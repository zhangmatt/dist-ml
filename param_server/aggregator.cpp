#include "param_server/aggregator.h"

#include <fstream>
#include <stdexcept>

namespace distml::ps {

GradientAggregator::GradientAggregator(std::size_t dimension,
                                       double learning_rate,
                                       int expected_workers,
                                       AggregationMode mode,
                                       int staleness_bound)
    : dimension_(dimension),
      learning_rate_(learning_rate),
      expected_workers_(expected_workers),
      mode_(mode),
      staleness_bound_(staleness_bound),
      weights_(dimension, 0.0) {
  if (dimension == 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  if (expected_workers <= 0) {
    throw std::invalid_argument("expected_workers must be positive");
  }
}

WeightSnapshot GradientAggregator::pull() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return WeightSnapshot{global_step_, weights_};
}

PushResult GradientAggregator::push(int worker_id,
                                    std::int64_t worker_step,
                                    std::int64_t samples,
                                    const std::vector<double>& gradient) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_id < 0) {
    return {"ERROR", global_step_, static_cast<int>(pending_.size()),
            "worker_id must be non-negative"};
  }
  if (samples <= 0) {
    return {"ERROR", global_step_, static_cast<int>(pending_.size()),
            "samples must be positive"};
  }
  if (gradient.size() != dimension_) {
    return {"ERROR", global_step_, static_cast<int>(pending_.size()),
            "gradient dimension mismatch"};
  }

  if (mode_ == AggregationMode::kAsync) {
    if (worker_step + staleness_bound_ < global_step_) {
      return {"STALE", global_step_, 0, "gradient exceeded staleness bound"};
    }
    std::vector<double> gradient_sum(dimension_, 0.0);
    for (std::size_t i = 0; i < dimension_; ++i) {
      gradient_sum[i] = gradient[i] * static_cast<double>(samples);
    }
    apply_weighted_gradient_locked(gradient_sum, samples);
    return {"UPDATED", global_step_, 0, "async gradient applied"};
  }

  if (worker_step != global_step_) {
    return {"STALE", global_step_, static_cast<int>(pending_.size()),
            "sync gradient was computed from a non-current step"};
  }
  if (pending_.find(worker_id) != pending_.end()) {
    return {"ERROR", global_step_, static_cast<int>(pending_.size()),
            "duplicate gradient for worker at current step"};
  }

  auto& entry = pending_[worker_id];
  entry.samples = samples;
  entry.gradient_sum.assign(dimension_, 0.0);
  for (std::size_t i = 0; i < dimension_; ++i) {
    entry.gradient_sum[i] = gradient[i] * static_cast<double>(samples);
  }

  if (static_cast<int>(pending_.size()) < expected_workers_) {
    return {"QUEUED", global_step_, static_cast<int>(pending_.size()),
            "waiting for remaining workers"};
  }

  std::vector<double> accumulated(dimension_, 0.0);
  std::int64_t total_samples = 0;
  for (const auto& [_, pending] : pending_) {
    total_samples += pending.samples;
    for (std::size_t i = 0; i < dimension_; ++i) {
      accumulated[i] += pending.gradient_sum[i];
    }
  }
  pending_.clear();
  apply_weighted_gradient_locked(accumulated, total_samples);
  return {"UPDATED", global_step_, 0, "sync gradient batch applied"};
}

bool GradientAggregator::save_weights(const std::string& path, std::string* error) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream output(path);
  if (!output) {
    if (error != nullptr) {
      *error = "failed to open " + path;
    }
    return false;
  }
  output.precision(17);
  output << weights_.size() << '\n';
  for (std::size_t i = 0; i < weights_.size(); ++i) {
    if (i != 0) {
      output << ' ';
    }
    output << weights_[i];
  }
  output << '\n';
  output << "step " << global_step_ << '\n';
  return true;
}

void GradientAggregator::apply_weighted_gradient_locked(
    const std::vector<double>& gradient_sum, std::int64_t samples) {
  const double denom = static_cast<double>(samples);
  for (std::size_t i = 0; i < dimension_; ++i) {
    weights_[i] -= learning_rate_ * (gradient_sum[i] / denom);
  }
  ++global_step_;
}

AggregationMode parse_mode(const std::string& mode) {
  if (mode == "sync") {
    return AggregationMode::kSync;
  }
  if (mode == "async") {
    return AggregationMode::kAsync;
  }
  throw std::invalid_argument("mode must be sync or async");
}

std::string mode_name(AggregationMode mode) {
  return mode == AggregationMode::kSync ? "sync" : "async";
}

}  // namespace distml::ps
