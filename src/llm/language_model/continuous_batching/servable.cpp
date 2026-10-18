//*****************************************************************************
// Copyright 2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../../logging.hpp"
#include "../../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../http_payload.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../apis/openai_completions.hpp"
#include "../../text_utils.hpp"
#include "llm_executor.hpp"
#include "servable.hpp"

namespace ovms {

void ContinuousBatchingServable::logPerfMetrics(ov::genai::PerfMetrics& perfMetrics) {
    const size_t inputTokenCount = perfMetrics.get_num_input_tokens();
    const size_t outputTokenCount = perfMetrics.get_num_generated_tokens();
    // GenerationHandle metrics are scoped to this request; TTFT contains one sample.
    const double ttftMs = perfMetrics.get_ttft().mean;
    const double prefillSpeedTps = calculatePrefillSpeed(inputTokenCount, ttftMs);

    SPDLOG_LOGGER_DEBUG(
        llm_calculator_logger,
        "Request processing metrics | input_token_count: {} | output_token_count: {} | total_token_count: {} | ttft_ms: {:.3f} | prefill_speed_tps: {:.3f}",
        inputTokenCount,
        outputTokenCount,
        inputTokenCount + outputTokenCount,
        ttftMs,
        prefillSpeedTps);
}

void ContinuousBatchingServable::notifyExecutorThread() {
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Notifying executor thread");
    if (properties->llmExecutorWrapper == nullptr) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "LLMExecutorWrapper is not initialized");
        return;
    }
    properties->llmExecutorWrapper->notifyNewRequestArrived();
}

absl::Status ContinuousBatchingServable::addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) {
    // Additional validation for big prompt and setting without dynamic split fuse (GenAI checks it during scheduling which is too late for us)
    if (executionContext->inputRequest.inputIds.get_size() > properties->schedulerConfig.max_num_batched_tokens && properties->schedulerConfig.dynamic_split_fuse == false) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received request with more tokens than max_num_batch_tokens {} > {}. Without dynamic split fuse on, such request is invalid",
            executionContext->inputRequest.inputIds.get_size(), properties->schedulerConfig.max_num_batched_tokens);
        return absl::InvalidArgumentError("Input length exceeds pipeline capabilities: " + std::to_string(executionContext->inputRequest.inputIds.get_size()) +
                                          " > " + std::to_string(properties->schedulerConfig.max_num_batched_tokens));
    }
    uint64_t requestId = currentRequestId++;
    try {
        executionContext->generationHandle = properties->llmExecutorWrapper->executeSynchronously(
            [this, requestId, &executionContext](ov::genai::ContinuousBatchingPipeline* pipe) {
                return pipe->add_request(requestId,
                    executionContext->inputRequest.inputIds,
                    executionContext->inputRequest.generationConfig);
            });
        properties->llmExecutorWrapper->registerActiveHandle(executionContext->generationHandle);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to add request {} to the pipeline: {}.", requestId, e.what());
        return absl::InternalError("Failed to add request to the pipeline");
    }
    return absl::OkStatus();
}

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> ContinuousBatchingServable::createExecutionContext() {
    return std::make_shared<ContinuousBatchingServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> ContinuousBatchingServable::getProperties() {
    return properties;
}

absl::Status ContinuousBatchingServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    auto status = addRequestToPipeline(cbExecutionContext);
    if (!status.ok()) {
        return status;
    }

    cbExecutionContext->payload.client->registerDisconnectionCallback([genHandle = cbExecutionContext->generationHandle]() {
        genHandle->stop();
    });
    notifyExecutorThread();

    return absl::OkStatus();
}

// This should probably be moved to GenAI
static ov::genai::GenerationOutput prepareEmptyStopReasonOutput() {
    static ov::genai::GenerationOutput out = {
        std::vector<int64_t>(),  // generated_ids
        std::vector<float>(),    // generated_log_probs
        0.0f,                    // score
        ov::genai::GenerationFinishReason::STOP};
    return out;
}

namespace {
// Non-blocking equivalent of GenAI's GenerationHandleImpl::read_all().
// read_all() blocks in the stream queue waiting for the next token chunk, so when the
// step loop dies (e.g. LA cache capacity exhaustion on GPU) an in-flight unary request
// would hang forever. Here we poll, merge partial outputs, and bail out the moment the
// pipeline is marked as failed, draining whatever was already produced.
void mergePartialOutputs(std::unordered_map<uint64_t, ov::genai::GenerationOutput>& partial,
    const ov::genai::GenerationOutputs& iteration) {
    for (auto& entry : iteration) {
        auto partialIter = partial.find(entry.first);
        if (partialIter == partial.end()) {
            partial.emplace(entry.first, entry.second);
        } else {
            auto generatedLen = entry.second.generated_ids.size();
            for (size_t i = 0; i < generatedLen; ++i) {
                partialIter->second.generated_ids.push_back(entry.second.generated_ids[i]);
                partialIter->second.generated_log_probs.push_back(entry.second.generated_log_probs[i]);
            }
            partialIter->second.score = entry.second.score;
            partialIter->second.finish_reason = entry.second.finish_reason;
            if (!entry.second.intermediate_hidden_states.empty()) {
                partialIter->second.intermediate_hidden_states = entry.second.intermediate_hidden_states;
            }
        }
    }
}

std::vector<ov::genai::GenerationOutput> collectOutputsNonBlocking(const ov::genai::GenerationHandle& handle,
    LLMExecutorWrapper* executorWrapper) {
    std::unordered_map<uint64_t, ov::genai::GenerationOutput> partial;
    for (;;) {
        try {
            if (executorWrapper->isPipelineFailed()) {
                break;
            }
            if (handle->get_status() != ov::genai::GenerationStatus::RUNNING) {
                break;
            }
            if (handle->can_read()) {
                mergePartialOutputs(partial, handle->read());
                continue;
            }
        } catch (...) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Drain any chunk(s) buffered after the exit condition (final result or failure).
    for (;;) {
        try {
            if (!handle->can_read()) {
                break;
            }
            mergePartialOutputs(partial, handle->read());
        } catch (...) {
            break;
        }
    }
    std::vector<ov::genai::GenerationOutput> results;
    results.reserve(partial.size());
    for (auto& entry : partial) {
        results.emplace_back(std::move(entry.second));
    }
    return results;
}
}  // namespace

absl::Status ContinuousBatchingServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    cbExecutionContext->generationOutputs = collectOutputsNonBlocking(cbExecutionContext->generationHandle, properties->llmExecutorWrapper.get());
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::STOP) {
        return absl::CancelledError();
    }
    if (cbExecutionContext->generationOutputs.size() == 0) {
        if (properties->llmExecutorWrapper->isPipelineFailed()) {
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "LLM generation pipeline failed before any output was produced.");
            return absl::InternalError("LLM generation pipeline failed");
        }
        cbExecutionContext->generationOutputs = {prepareEmptyStopReasonOutput()};
    }
    executionContext->perfMetrics = std::make_unique<ov::genai::PerfMetrics>(cbExecutionContext->generationHandle->get_perf_metrics());
    return absl::OkStatus();
}

absl::Status ContinuousBatchingServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingServableExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    // Streaming scenario
    // Each iteration is single execution of Process() method in the calculator
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::STOP) {
        return absl::CancelledError();
    }
    // A dead step loop would leave the stream queue empty forever and a blocking read()
    // would hang the request; fail fast instead and let the client terminate the stream.
    if (properties->llmExecutorWrapper->isPipelineFailed()) {
        return absl::InternalError("LLM generation pipeline failed");
    }

    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::RUNNING || cbExecutionContext->generationHandle->can_read()) {
        // Subsequent iteration
        OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
        ov::genai::GenerationOutputs generationOutputs = cbExecutionContext->generationHandle->read();
        RET_CHECK(generationOutputs.size() <= 1);  // TODO: Support multiple generations
        if (generationOutputs.size() == 0) {
            cbExecutionContext->generationOutputs = {prepareEmptyStopReasonOutput()};
        } else {
            cbExecutionContext->generationOutputs = {generationOutputs.begin()->second};
        }
        if (cbExecutionContext->generationOutputs[0].finish_reason != ov::genai::GenerationFinishReason::NONE) {
            executionContext->perfMetrics = std::make_unique<ov::genai::PerfMetrics>(cbExecutionContext->generationHandle->get_perf_metrics());
        }
    }
    return absl::OkStatus();
}

absl::Status ContinuousBatchingServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto status = GenAiServable::prepareCompleteResponse(executionContext);
    if (status.ok() && executionContext->perfMetrics && llm_calculator_logger->should_log(spdlog::level::debug)) {
        logPerfMetrics(*executionContext->perfMetrics);
    }
    return status;
}

absl::Status ContinuousBatchingServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto status = GenAiServable::preparePartialResponse(executionContext);
    if (status.ok() &&
        !executionContext->sendLoopbackSignal &&
        executionContext->perfMetrics &&
        llm_calculator_logger->should_log(spdlog::level::debug)) {
        logPerfMetrics(*executionContext->perfMetrics);
    }
    return status;
}

}  // namespace ovms
