#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "serving/autoscaler.h"
#include "serving/batcher.h"
#include "serving/inference_server.h"

namespace {

struct Args {
  std::string host = "127.0.0.1";
  int port = 50052;
  std::string weights_path = "build/model.weights";
  int max_batch_size = 16;
  int max_delay_ms = 5;
  int min_replicas = 1;
  int max_replicas = 8;
  double p99_target_ms = 25.0;
};

void usage(const char* program) {
  std::cerr << "usage: " << program
            << " --weights path [--host 127.0.0.1] [--port 50052]"
               " [--max-batch-size 16] [--max-delay-ms 5]"
               " [--min-replicas 1] [--max-replicas 8] [--p99-target-ms 25]\n";
}

bool parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      usage(argv[0]);
      return false;
    }
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << key << '\n';
      return false;
    }
    const std::string value = argv[++i];
    if (key == "--host") {
      args->host = value;
    } else if (key == "--port") {
      args->port = std::atoi(value.c_str());
    } else if (key == "--weights") {
      args->weights_path = value;
    } else if (key == "--max-batch-size") {
      args->max_batch_size = std::atoi(value.c_str());
    } else if (key == "--max-delay-ms") {
      args->max_delay_ms = std::atoi(value.c_str());
    } else if (key == "--min-replicas") {
      args->min_replicas = std::atoi(value.c_str());
    } else if (key == "--max-replicas") {
      args->max_replicas = std::atoi(value.c_str());
    } else if (key == "--p99-target-ms") {
      args->p99_target_ms = std::atof(value.c_str());
    } else {
      std::cerr << "unknown argument: " << key << '\n';
      return false;
    }
  }
  return args->port > 0 && args->max_batch_size > 0 && args->max_delay_ms >= 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    usage(argv[0]);
    return 2;
  }

  try {
    const auto model = distml::serving::load_weights(args.weights_path);
    distml::serving::AutoscalerConfig autoscaler_config;
    autoscaler_config.min_replicas = args.min_replicas;
    autoscaler_config.max_replicas = args.max_replicas;
    autoscaler_config.p99_latency_target_ms = args.p99_target_ms;

    auto batcher = std::make_shared<distml::serving::DynamicBatcher>(
        model.weights, model.step, args.max_batch_size, args.max_delay_ms,
        distml::serving::Autoscaler(autoscaler_config));
    distml::serving::InferenceServer server(args.host, args.port, batcher);
    server.run();
  } catch (const std::exception& ex) {
    std::cerr << "inference_server error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
