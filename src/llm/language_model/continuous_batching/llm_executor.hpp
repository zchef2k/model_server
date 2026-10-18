//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <openvino/genai/continuous_batching_pipeline.hpp>

#include "../../../logging.hpp"
#include "../../../profiler.hpp"

namespace ovms {
struct LLMExecutor {
    bool isDynamicKVCache;
    // For logging purposes we could have more information about graph and node here
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe = nullptr;

    LLMExecutor(std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe, bool isDynamicKVCacheSet = false) {
        this->pipe = std::move(pipe);
        this->isDynamicKVCache = isDynamicKVCacheSet;
    }

    bool hasRequests() {
        return pipe && pipe->has_non_finished_requests();
    }

    void step() {
        OVMS_PROFILE_FUNCTION();
        if (pipe) {
            pipe->step();
        }
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal) {
        waitForRequests(receivedEndSignal, nullptr);
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal, const std::function<bool()>& isTaskPending) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this, receivedEndSignal, &isTaskPending] {
            return (pipe && pipe->has_non_finished_requests()) || *receivedEndSignal ||
                (isTaskPending && isTaskPending());
        });
    }

    void notify() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_one();
    }

    std::string formatCacheInfo(float cacheUsage, size_t cacheBytes, bool isCacheDynamic) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        if (isCacheDynamic) {
            oss << "type: dynamic, cache usage: " << cacheUsage << "% of " << formatBytes(cacheBytes);
        } else {
            oss << "type: static, cache usage: " << cacheUsage << "% of " << formatBytes(cacheBytes);
        }

        return oss.str();
    }

    std::string formatBytes(size_t bytes) {
        const double KB = 1024.0;
        const double MB = KB * 1024.0;
        const double GB = MB * 1024.0;
        const double TB = GB * 1024.0;

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);

        if (bytes >= TB)
            oss << (bytes / TB) << " TB";
        else if (bytes >= GB)
            oss << (bytes / GB) << " GB";
        else if (bytes >= MB)
            oss << (bytes / MB) << " MB";
        else if (bytes >= KB)
            oss << (bytes / KB) << " KB";
        else
            oss << bytes << " B";

        return oss.str();
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    void printMetrics() {
        ov::genai::PipelineMetrics metrics = pipe->get_metrics();
        SPDLOG_LOGGER_INFO(llm_executor_logger, "All requests: {}; Scheduled requests: {}; Cache {};",
            metrics.requests, metrics.scheduled_requests, formatCacheInfo(metrics.cache_usage, metrics.kv_cache_size_in_bytes, this->isDynamicKVCache));
    }
};
#pragma GCC diagnostic pop

class LLMExecutorWrapper {
    LLMExecutor llmExecutor;
    std::thread llmExecutorThread;
    std::atomic<bool> finishExecutorThread = false;
    std::function<std::shared_ptr<ov::genai::ContinuousBatchingPipeline>()> pipeFactory;
    std::function<void(const std::shared_ptr<ov::genai::ContinuousBatchingPipeline>&)> onPipelineCreated;
    std::promise<void> pipelineReady;
    std::future<void> readyFuture = pipelineReady.get_future();
    std::mutex taskMutex;
    std::deque<std::function<void(ov::genai::ContinuousBatchingPipeline*)>> pendingTasks;
    std::atomic<bool> stepFailed = false;
    std::vector<ov::genai::GenerationHandle> activeHandles;
    std::function<void()> onStepError;

    void drainTasks() {
        while (true) {
            std::function<void(ov::genai::ContinuousBatchingPipeline*)> task;
            {
                std::lock_guard<std::mutex> lock(taskMutex);
                if (pendingTasks.empty()) {
                    return;
                }
                task = std::move(pendingTasks.front());
                pendingTasks.pop_front();
            }
            task(llmExecutor.pipe.get());
        }
    }

    static void run(LLMExecutorWrapper* self) {
        try {
            if (self->pipeFactory) {
                auto pipe = self->pipeFactory();
                if (self->onPipelineCreated) {
                    self->onPipelineCreated(pipe);
                }
                self->llmExecutor.pipe = std::move(pipe);
            }
            self->pipelineReady.set_value();
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error during LLM pipeline initialization on executor thread: {}.", e.what());
            self->pipelineReady.set_exception(std::current_exception());
            self->finishExecutorThread = true;
            self->llmExecutor.notify();
            return;
        } catch (...) {
            self->pipelineReady.set_exception(std::current_exception());
            self->finishExecutorThread = true;
            self->llmExecutor.notify();
            return;
        }
        const uint8_t printMetricsEveryNumberOfSteps = 10;
        uint8_t stepCounter = 0;
        while (!(self->finishExecutorThread)) {
            try {
                self->drainTasks();
                if (stepCounter == printMetricsEveryNumberOfSteps) {
                    self->llmExecutor.printMetrics();
                    stepCounter = 0;
                }
                if (self->llmExecutor.hasRequests() && !self->stepFailed) {
                    stepCounter++;
                    self->llmExecutor.step();
                } else if (self->stepFailed) {
                    // Pipeline is corrupt (GPU residual state) and cannot be stepped anymore.
                    // waitForRequests would spin forever because GenAI's own predicate keeps
                    // observing the unfinished request, so sleep instead. Queued new-request
                    // tasks are still drained at the top of every iteration (they fail fast).
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                } else {
                    SPDLOG_LOGGER_INFO(llm_executor_logger, "All requests: {}; Scheduled requests: {};", 0, 0);
                    self->llmExecutor.waitForRequests(&(self->finishExecutorThread), [self] {
                        std::lock_guard<std::mutex> lock(self->taskMutex);
                        return !self->pendingTasks.empty();
                    });
                }
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_ERROR(llm_executor_logger,
                    "LLM pipeline step failed. Stopping pipeline stepping for this servable to prevent further GPU state corruption: {}.",
                    e.what());
                self->stepFailed = true;
                self->cancelActiveHandles();
                if (self->onStepError) {
                    try {
                        self->onStepError();
                    } catch (...) {
                    }
                }
            }
        }
    }

public:
    LLMExecutorWrapper(std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe, bool isDynamicKVCache = false) :
        llmExecutor(std::move(pipe), isDynamicKVCache) {
        llmExecutorThread = std::thread(LLMExecutorWrapper::run, this);
    }

    LLMExecutorWrapper(std::function<std::shared_ptr<ov::genai::ContinuousBatchingPipeline>()> pipeFactory,
        std::function<void(const std::shared_ptr<ov::genai::ContinuousBatchingPipeline>&)> onPipelineCreated,
        bool isDynamicKVCache = false) :
        llmExecutor(nullptr, isDynamicKVCache),
        pipeFactory(std::move(pipeFactory)),
        onPipelineCreated(std::move(onPipelineCreated)) {
        llmExecutorThread = std::thread(LLMExecutorWrapper::run, this);
    }

    ~LLMExecutorWrapper() {
        finishExecutorThread = true;
        llmExecutor.notify();
        llmExecutorThread.join();
    }

    std::future<void>& getReadyFuture() {
        return readyFuture;
    }

    void notifyNewRequestArrived() {
        llmExecutor.notify();
    }

    template <typename F>
    std::invoke_result_t<F&, ov::genai::ContinuousBatchingPipeline*> executeSynchronously(F&& task) {
        using T = std::invoke_result_t<F&, ov::genai::ContinuousBatchingPipeline*>;
        std::promise<T> promise;
        std::future<T> future = promise.get_future();
        auto wrapped = [task = std::move(task), &promise](ov::genai::ContinuousBatchingPipeline* pipe) mutable {
            try {
                if constexpr (std::is_void_v<T>) {
                    task(pipe);
                    promise.set_value();
                } else {
                    promise.set_value(task(pipe));
                }
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        };
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            pendingTasks.push_back(std::move(wrapped));
        }
        llmExecutor.notify();
        return future.get();
    }

    bool isPipelineFailed() const {
        return stepFailed.load();
    }

    void setOnStepError(std::function<void()> callback) {
        onStepError = std::move(callback);
    }

    void registerActiveHandle(const ov::genai::GenerationHandle& handle) {
        std::lock_guard<std::mutex> lock(taskMutex);
        // Prune handles of already completed requests so the registry does not grow unbounded.
        activeHandles.erase(
            std::remove_if(activeHandles.begin(), activeHandles.end(),
                [](const ov::genai::GenerationHandle& h) {
                    try {
                        return h->get_status() != ov::genai::GenerationStatus::RUNNING;
                    } catch (...) {
                        return false;
                    }
                }),
            activeHandles.end());
        activeHandles.push_back(handle);
    }

    void cancelActiveHandles() {
        std::vector<ov::genai::GenerationHandle> handles;
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            handles.swap(activeHandles);
        }
        for (auto& handle : handles) {
            try {
                handle->cancel();
            } catch (...) {
            }
        }
    }
};

}  // namespace ovms
