#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace distml::ps {

enum class AggregationMode {
  kSync,
  kAsync,
};

struct PushResult {
  std::string status;
  std::int64_t global_step = 0;
  int pending_workers = 0;
  std::string message;
};

struct WeightSnapshot {
  std::int64_t step = 0;
  std::vector<double> weights;
};

class GradientAggregator {
 public:
  GradientAggregator(std::size_t dimension,
                     double learning_rate,
                     int expected_workers,
                     AggregationMode mode,
                     int staleness_bound = 2);

  WeightSnapshot pull() const;
  PushResult push(int worker_id,
                  std::int64_t worker_step,
                  std::int64_t samples,
                  const std::vector<double>& gradient);
  bool save_weights(const std::string& path, std::string* error) const;

  std::size_t dimension() const { return dimension_; }
  double learning_rate() const { return learning_rate_; }
  int expected_workers() const { return expected_workers_; }
  AggregationMode mode() const { return mode_; }

 private:
  struct PendingGradient {
    std::vector<double> gradient_sum;
    std::int64_t samples = 0;
  };

  void apply_weighted_gradient_locked(const std::vector<double>& gradient_sum,
                                      std::int64_t samples);

  std::size_t dimension_;
  double learning_rate_;
  int expected_workers_;
  AggregationMode mode_;
  int staleness_bound_;

  mutable std::mutex mutex_;
  std::int64_t global_step_ = 0;
  std::vector<double> weights_;
  std::unordered_map<int, PendingGradient> pending_;
};

AggregationMode parse_mode(const std::string& mode);
std::string mode_name(AggregationMode mode);

}  // namespace distml::ps
