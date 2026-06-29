#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "param_server/aggregator.h"
#include "serving/autoscaler.h"
#include "serving/batcher.h"

namespace {

bool near(double actual, double expected, double eps = 1e-9) {
  return std::fabs(actual - expected) <= eps;
}

void test_sync_aggregation() {
  distml::ps::GradientAggregator agg(2, 1.0, 2, distml::ps::AggregationMode::kSync);
  auto first = agg.push(0, 0, 2, std::vector<double>{2.0, 0.0});
  assert(first.status == "QUEUED");
  assert(first.global_step == 0);

  auto second = agg.push(1, 0, 2, std::vector<double>{0.0, 2.0});
  assert(second.status == "UPDATED");
  assert(second.global_step == 1);

  auto snapshot = agg.pull();
  assert(snapshot.step == 1);
  assert(near(snapshot.weights[0], -1.0));
  assert(near(snapshot.weights[1], -1.0));
}

void test_async_aggregation() {
  distml::ps::GradientAggregator agg(2, 0.5, 2, distml::ps::AggregationMode::kAsync);
  auto result = agg.push(0, 0, 1, std::vector<double>{2.0, -4.0});
  assert(result.status == "UPDATED");

  auto snapshot = agg.pull();
  assert(snapshot.step == 1);
  assert(near(snapshot.weights[0], -1.0));
  assert(near(snapshot.weights[1], 2.0));
}

void test_autoscaler() {
  distml::serving::AutoscalerConfig config;
  config.min_replicas = 1;
  config.max_replicas = 4;
  config.queue_per_replica = 4;
  config.p99_latency_target_ms = 10.0;
  distml::serving::Autoscaler scaler(config);

  assert(scaler.observe(0, 0.0) == 1);
  assert(scaler.observe(12, 1.0) == 3);
  assert(scaler.observe(12, 50.0) == 4);
  assert(scaler.observe(0, 1.0) == 3);
}

void test_batcher_prediction() {
  distml::serving::AutoscalerConfig config;
  distml::serving::DynamicBatcher batcher(
      std::vector<double>{1.0, -0.5, 0.0},
      7,
      4,
      1,
      distml::serving::Autoscaler(config));

  auto prediction = batcher.predict(std::vector<double>{1.0, 1.0});
  assert(prediction.model_version == 7);
  assert(prediction.label == 1);
  assert(near(prediction.probability, 1.0 / (1.0 + std::exp(-0.5)), 1e-6));

  auto metrics = batcher.metrics();
  assert(metrics.processed_requests == 1);
  assert(metrics.processed_batches == 1);
  batcher.stop();
}

}  // namespace

int main() {
  test_sync_aggregation();
  test_async_aggregation();
  test_autoscaler();
  test_batcher_prediction();
  std::cout << "cpp_core_test passed\n";
  return 0;
}
