#pragma once

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

class BoundedEventQueue {
public:
    BoundedEventQueue(size_t max_size = 10000) : max_size_(max_size) {}

    void push(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            // Drop oldest to prevent memory exhaustion (OOM)
            queue_.pop();
        }
        queue_.push(event);
        cond_.notify_one();
    }

    bool pop(std::string& event, const std::atomic<bool>& keep_running) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty()) {
            if (!keep_running) {
                return false;
            }
            cond_.wait_for(lock, std::chrono::milliseconds(100));
        }
        event = queue_.front();
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<std::string> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    size_t max_size_;
};
