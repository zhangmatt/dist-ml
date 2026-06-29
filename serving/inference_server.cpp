#include "serving/inference_server.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "common/protocol.h"
#include "common/socket_utils.h"

namespace distml::serving {

using distml::protocol::format_double;
using distml::protocol::parse_double;
using distml::protocol::parse_int;
using distml::protocol::split_ws;

LoadedModel load_weights(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open model weights: " + path);
  }
  std::size_t dim = 0;
  input >> dim;
  if (dim < 2) {
    throw std::runtime_error("model weights file has invalid dimension");
  }
  LoadedModel model;
  model.weights.assign(dim, 0.0);
  for (std::size_t i = 0; i < dim; ++i) {
    if (!(input >> model.weights[i])) {
      throw std::runtime_error("model weights file ended early");
    }
  }
  std::string step_label;
  if (input >> step_label >> model.step && step_label != "step") {
    model.step = 0;
  }
  return model;
}

InferenceServer::InferenceServer(std::string host,
                                 int port,
                                 std::shared_ptr<DynamicBatcher> batcher)
    : host_(std::move(host)), port_(port), batcher_(std::move(batcher)) {}

void InferenceServer::run() {
  distml::net::TcpServer server(host_, port_);
  {
    std::lock_guard<std::mutex> lock(server_mutex_);
    server_ = &server;
  }
  std::cerr << "inference_server listening on " << host_ << ':' << port_ << '\n';
  while (!stopping_.load()) {
    int client_fd = server.accept_one();
    if (client_fd < 0) {
      if (stopping_.load()) {
        break;
      }
      continue;
    }
    std::thread(&InferenceServer::handle_client, this, client_fd).detach();
  }
  {
    std::lock_guard<std::mutex> lock(server_mutex_);
    server_ = nullptr;
  }
}

void InferenceServer::stop() {
  stopping_.store(true);
  {
    std::lock_guard<std::mutex> lock(server_mutex_);
    if (server_ != nullptr) {
      server_->stop();
    }
  }
  if (batcher_) {
    batcher_->stop();
  }
}

void InferenceServer::handle_client(int client_fd) {
  std::string line;
  if (distml::net::read_line(client_fd, &line)) {
    const std::string response = handle_command(line) + "\n";
    distml::net::write_all(client_fd, response);
  }
  distml::net::close_fd(client_fd);
}

std::string InferenceServer::handle_command(const std::string& line) {
  const auto tokens = split_ws(line);
  if (tokens.empty()) {
    return "ERROR empty command";
  }

  if (tokens[0] == "HEALTH") {
    return "OK";
  }

  if (tokens[0] == "PREDICT") {
    if (tokens.size() < 3) {
      return "ERROR PREDICT requires dim and features";
    }
    int dim = 0;
    if (!parse_int(tokens[1], &dim) || dim < 0 ||
        tokens.size() != static_cast<std::size_t>(2 + dim)) {
      return "ERROR invalid PREDICT arguments";
    }
    std::vector<double> features(static_cast<std::size_t>(dim), 0.0);
    for (int i = 0; i < dim; ++i) {
      if (!parse_double(tokens[2 + i], &features[static_cast<std::size_t>(i)])) {
        return "ERROR invalid feature value";
      }
    }
    try {
      const Prediction prediction = batcher_->predict(features);
      return "PREDICTION " + format_double(prediction.probability) + " " +
             std::to_string(prediction.label) + " " +
             std::to_string(prediction.model_version);
    } catch (const std::exception& ex) {
      return std::string("ERROR ") + ex.what();
    }
  }

  if (tokens[0] == "METRICS") {
    const ServingMetrics m = batcher_->metrics();
    return "METRICS " + std::to_string(m.processed_requests) + " " +
           std::to_string(m.processed_batches) + " " +
           std::to_string(m.queue_depth) + " " +
           std::to_string(m.desired_replicas) + " " +
           format_double(m.average_latency_ms) + " " + format_double(m.p99_latency_ms);
  }

  if (tokens[0] == "STOP") {
    stop();
    return "BYE";
  }

  return "ERROR unknown command";
}

}  // namespace distml::serving
