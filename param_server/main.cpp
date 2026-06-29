#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "param_server/aggregator.h"
#include "param_server/ps_server.h"

namespace {

struct Args {
  std::string host = "127.0.0.1";
  int port = 50051;
  int dim = 0;
  double lr = 0.1;
  int workers = 1;
  std::string mode = "sync";
  int staleness_bound = 2;
};

void usage(const char* program) {
  std::cerr << "usage: " << program
            << " --dim N [--host 127.0.0.1] [--port 50051] [--lr 0.1]"
               " [--workers 2] [--mode sync|async] [--staleness-bound 2]\n";
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
    } else if (key == "--dim") {
      args->dim = std::atoi(value.c_str());
    } else if (key == "--lr") {
      args->lr = std::atof(value.c_str());
    } else if (key == "--workers") {
      args->workers = std::atoi(value.c_str());
    } else if (key == "--mode") {
      args->mode = value;
    } else if (key == "--staleness-bound") {
      args->staleness_bound = std::atoi(value.c_str());
    } else {
      std::cerr << "unknown argument: " << key << '\n';
      return false;
    }
  }
  return args->dim > 0 && args->port > 0 && args->workers > 0 && args->lr > 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    usage(argv[0]);
    return 2;
  }

  try {
    auto aggregator = std::make_shared<distml::ps::GradientAggregator>(
        static_cast<std::size_t>(args.dim), args.lr, args.workers,
        distml::ps::parse_mode(args.mode), args.staleness_bound);
    distml::ps::ParameterServer server(args.host, args.port, aggregator);
    server.run();
  } catch (const std::exception& ex) {
    std::cerr << "parameter_server error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
