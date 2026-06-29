#include "serving/autoscaler.h"

#include <algorithm>

namespace distml::serving {

Autoscaler::Autoscaler(AutoscalerConfig config)
    : config_(config), desired_replicas_(config.min_replicas) {
  if (config_.min_replicas < 1) {
    config_.min_replicas = 1;
  }
  if (config_.max_replicas < config_.min_replicas) {
    config_.max_replicas = config_.min_replicas;
  }
  if (config_.queue_per_replica < 1) {
    config_.queue_per_replica = 1;
  }
  desired_replicas_ = config_.min_replicas;
}

int Autoscaler::observe(int queue_depth, double p99_latency_ms) {
  const int queue_target =
      std::max(config_.min_replicas,
               (queue_depth + config_.queue_per_replica - 1) / config_.queue_per_replica);

  int next = std::max(desired_replicas_, queue_target);
  if (p99_latency_ms > config_.p99_latency_target_ms && queue_depth > 0) {
    next = std::max(next, desired_replicas_ + 1);
  }
  if (queue_depth == 0 && p99_latency_ms < config_.p99_latency_target_ms * 0.5) {
    next = desired_replicas_ - 1;
  }

  desired_replicas_ = std::clamp(next, config_.min_replicas, config_.max_replicas);
  return desired_replicas_;
}

}  // namespace distml::serving
