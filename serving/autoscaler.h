#pragma once

namespace distml::serving {

struct AutoscalerConfig {
  int min_replicas = 1;
  int max_replicas = 8;
  int queue_per_replica = 8;
  double p99_latency_target_ms = 25.0;
};

class Autoscaler {
 public:
  explicit Autoscaler(AutoscalerConfig config);

  int observe(int queue_depth, double p99_latency_ms);
  int desired_replicas() const { return desired_replicas_; }

 private:
  AutoscalerConfig config_;
  int desired_replicas_;
};

}  // namespace distml::serving
