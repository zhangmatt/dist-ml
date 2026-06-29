#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "serving/autoscaler.h"

namespace distml::serving {

struct Prediction {
  double probability = 0.0;
  int label = 0;
  std::int64_t model_version = 0;
};

struct ServingMetrics {
  std::int64_t processed_requests = 0;
  std::int64_t processed_batches = 0;
  int queue_depth = 0;
  int desired_replicas = 1;
  double average_latency_ms = 0.0;
  double p99_latency_ms = 0.0;
};

class DynamicBatcher {
 public:
  DynamicBatcher(std::vector<double> weights,
                 std::int64_t model_version,
                 int max_batch_size,
                 int max_delay_ms,
                 Autoscaler autoscaler);
  ~DynamicBatcher();

  DynamicBatcher(const DynamicBatcher&) = delete;
  DynamicBatcher& operator=(const DynamicBatcher&) = delete;

  Prediction predict(const std::vector<double>& features);
  ServingMetrics metrics();
  void stop();

 private:
  struct WorkItem {
    std::vector<double> features;
    std::chrono::steady_clock::time_point enqueued_at;
    std::promise<Prediction> result;
  };

  void run();
  Prediction predict_one(const std::vector<double>& features) const;
  void record_latency(double latency_ms);

  std::vector<double> weights_;
  std::int64_t model_version_;
  int max_batch_size_;
  int max_delay_ms_;
  Autoscaler autoscaler_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
  std::deque<WorkItem> queue_;
  std::thread worker_;

  std::int64_t processed_requests_ = 0;
  std::int64_t processed_batches_ = 0;
  std::deque<double> recent_latencies_ms_;
};

}  // namespace distml::serving
