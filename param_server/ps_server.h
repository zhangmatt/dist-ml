#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "common/socket_utils.h"
#include "param_server/aggregator.h"

namespace distml::ps {

class ParameterServer {
 public:
  ParameterServer(std::string host,
                  int port,
                  std::shared_ptr<GradientAggregator> aggregator);

  void run();
  void stop();

 private:
  void handle_client(int client_fd);
  std::string handle_command(const std::string& line);

  std::string host_;
  int port_;
  std::shared_ptr<GradientAggregator> aggregator_;
  std::atomic<bool> stopping_{false};
  std::mutex server_mutex_;
  distml::net::TcpServer* server_ = nullptr;
};

}  // namespace distml::ps
