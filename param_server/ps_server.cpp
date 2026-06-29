#include "param_server/ps_server.h"

#include <iostream>
#include <thread>

#include "common/protocol.h"
#include "common/socket_utils.h"

namespace distml::ps {

using distml::protocol::join_doubles;
using distml::protocol::parse_double;
using distml::protocol::parse_i64;
using distml::protocol::parse_int;
using distml::protocol::split_ws;

ParameterServer::ParameterServer(std::string host,
                                 int port,
                                 std::shared_ptr<GradientAggregator> aggregator)
    : host_(std::move(host)), port_(port), aggregator_(std::move(aggregator)) {}

void ParameterServer::run() {
  distml::net::TcpServer server(host_, port_);
  {
    std::lock_guard<std::mutex> lock(server_mutex_);
    server_ = &server;
  }
  std::cerr << "parameter_server listening on " << host_ << ':' << port_
            << " mode=" << mode_name(aggregator_->mode())
            << " dim=" << aggregator_->dimension()
            << " workers=" << aggregator_->expected_workers() << '\n';

  while (!stopping_.load()) {
    int client_fd = server.accept_one();
    if (client_fd < 0) {
      if (stopping_.load()) {
        break;
      }
      continue;
    }
    std::thread(&ParameterServer::handle_client, this, client_fd).detach();
  }
  {
    std::lock_guard<std::mutex> lock(server_mutex_);
    server_ = nullptr;
  }
}

void ParameterServer::stop() {
  stopping_.store(true);
  std::lock_guard<std::mutex> lock(server_mutex_);
  if (server_ != nullptr) {
    server_->stop();
  }
}

void ParameterServer::handle_client(int client_fd) {
  std::string line;
  if (distml::net::read_line(client_fd, &line)) {
    const std::string response = handle_command(line) + "\n";
    distml::net::write_all(client_fd, response);
  }
  distml::net::close_fd(client_fd);
}

std::string ParameterServer::handle_command(const std::string& line) {
  const auto tokens = split_ws(line);
  if (tokens.empty()) {
    return "ERROR empty command";
  }

  if (tokens[0] == "HEALTH") {
    return "OK";
  }

  if (tokens[0] == "PULL") {
    const auto snapshot = aggregator_->pull();
    return "WEIGHTS " + std::to_string(snapshot.step) + " " +
           std::to_string(snapshot.weights.size()) + join_doubles(snapshot.weights);
  }

  if (tokens[0] == "PUSH") {
    if (tokens.size() < 6) {
      return "ERROR PUSH requires worker_id step samples dim gradients";
    }
    int worker_id = 0;
    std::int64_t worker_step = 0;
    std::int64_t samples = 0;
    int dim = 0;
    if (!parse_int(tokens[1], &worker_id) || !parse_i64(tokens[2], &worker_step) ||
        !parse_i64(tokens[3], &samples) || !parse_int(tokens[4], &dim) ||
        dim < 0 || tokens.size() != static_cast<std::size_t>(5 + dim)) {
      return "ERROR invalid PUSH arguments";
    }
    std::vector<double> gradient(static_cast<std::size_t>(dim), 0.0);
    for (int i = 0; i < dim; ++i) {
      if (!parse_double(tokens[5 + i], &gradient[static_cast<std::size_t>(i)])) {
        return "ERROR invalid gradient value";
      }
    }
    const auto result = aggregator_->push(worker_id, worker_step, samples, gradient);
    return result.status + " " + std::to_string(result.global_step) + " " +
           std::to_string(result.pending_workers) + " " + result.message;
  }

  if (tokens[0] == "SAVE") {
    if (tokens.size() != 2) {
      return "ERROR SAVE requires path";
    }
    std::string error;
    if (!aggregator_->save_weights(tokens[1], &error)) {
      return "ERROR " + error;
    }
    return "SAVED " + tokens[1];
  }

  if (tokens[0] == "STOP") {
    stop();
    return "BYE";
  }

  return "ERROR unknown command";
}

}  // namespace distml::ps
