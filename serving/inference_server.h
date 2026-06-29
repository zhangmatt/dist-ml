#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "common/socket_utils.h"
#include "serving/batcher.h"

namespace distml::serving {

struct LoadedModel {
  std::vector<double> weights;
  std::int64_t step = 0;
};

LoadedModel load_weights(const std::string& path);

class InferenceServer {
 public:
  InferenceServer(std::string host, int port, std::shared_ptr<DynamicBatcher> batcher);

  void run();
  void stop();

 private:
  void handle_client(int client_fd);
  std::string handle_command(const std::string& line);

  std::string host_;
  int port_;
  std::shared_ptr<DynamicBatcher> batcher_;
  std::atomic<bool> stopping_{false};
  std::mutex server_mutex_;
  distml::net::TcpServer* server_ = nullptr;
};

}  // namespace distml::serving
