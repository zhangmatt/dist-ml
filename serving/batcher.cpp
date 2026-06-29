#include "serving/batcher.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace distml::serving {

DynamicBatcher::DynamicBatcher(std::vector<double> weights,
                               std::int64_t model_version,
                               int max_batch_size,
                               int max_delay_ms,
                               Autoscaler autoscaler)
    : weights_(std::move(weights)),
      model_version_(model_version),
      max_batch_size_(max_batch_size),
      max_delay_ms_(max_delay_ms),
      autoscaler_(autoscaler) {
  if (weights_.size() < 2) {
    throw std::invalid_argument("serving weights need at least one feature and one bias");
  }
  if (max_batch_size_ <= 0) {
    throw std::invalid_argument("max_batch_size must be positive");
  }
  if (max_delay_ms_ < 0) {
    throw std::invalid_argument("max_delay_ms must be non-negative");
  }
  worker_ = std::thread(&DynamicBatcher::run, this);
}

DynamicBatcher::~DynamicBatcher() {
  stop();
}

Prediction DynamicBatcher::predict(const std::vector<double>& features) {
  WorkItem item;
  item.features = features;
  item.enqueued_at = std::chrono::steady_clock::now();
  auto future = item.result.get_future();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("batcher is stopping");
    }
    queue_.push_back(std::move(item));
  }
  cv_.notify_one();
  return future.get();
}

ServingMetrics DynamicBatcher::metrics() {
  std::lock_guard<std::mutex> lock(mutex_);
  ServingMetrics m;
  m.processed_requests = processed_requests_;
  m.processed_batches = processed_batches_;
  m.queue_depth = static_cast<int>(queue_.size());
  if (!recent_latencies_ms_.empty()) {
    m.average_latency_ms =
        std::accumulate(recent_latencies_ms_.begin(), recent_latencies_ms_.end(), 0.0) /
        static_cast<double>(recent_latencies_ms_.size());
    std::vector<double> sorted(recent_latencies_ms_.begin(), recent_latencies_ms_.end());
    std::sort(sorted.begin(), sorted.end());
    const std::size_t idx =
        std::min(sorted.size() - 1,
                 static_cast<std::size_t>(std::ceil(sorted.size() * 0.99) - 1));
    m.p99_latency_ms = sorted[idx];
  }
  m.desired_replicas = autoscaler_.observe(m.queue_depth, m.p99_latency_ms);
  return m;
}

void DynamicBatcher::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void DynamicBatcher::run() {
  while (true) {
    std::vector<WorkItem> batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) {
        return;
      }

      if (max_delay_ms_ > 0 && queue_.size() < static_cast<std::size_t>(max_batch_size_)) {
        cv_.wait_for(lock, std::chrono::milliseconds(max_delay_ms_), [&] {
          return stopping_ || queue_.size() >= static_cast<std::size_t>(max_batch_size_);
        });
      }

      const std::size_t take =
          std::min<std::size_t>(queue_.size(), static_cast<std::size_t>(max_batch_size_));
      batch.reserve(take);
      for (std::size_t i = 0; i < take; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }

    const auto completed_at = std::chrono::steady_clock::now();
    for (auto& item : batch) {
      Prediction prediction = predict_one(item.features);
      const double latency_ms = std::chrono::duration<double, std::milli>(
                                    completed_at - item.enqueued_at)
                                    .count();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++processed_requests_;
        record_latency(latency_ms);
      }
      item.result.set_value(prediction);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++processed_batches_;
    }
  }
}

Prediction DynamicBatcher::predict_one(const std::vector<double>& features) const {
  if (features.size() + 1 != weights_.size()) {
    throw std::runtime_error("feature dimension does not match model weights");
  }
  double logit = weights_.back();
  for (std::size_t i = 0; i < features.size(); ++i) {
    logit += features[i] * weights_[i];
  }
  const double probability = 1.0 / (1.0 + std::exp(-logit));
  return Prediction{probability, probability >= 0.5 ? 1 : 0, model_version_};
}

void DynamicBatcher::record_latency(double latency_ms) {
  recent_latencies_ms_.push_back(latency_ms);
  constexpr std::size_t kMaxSamples = 512;
  while (recent_latencies_ms_.size() > kMaxSamples) {
    recent_latencies_ms_.pop_front();
  }
}

}  // namespace distml::serving
