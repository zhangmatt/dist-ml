#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace distml::net {

int create_listening_socket(const std::string& host, int port, int backlog);
bool read_line(int fd, std::string* line);
bool write_all(int fd, const std::string& data);
void close_fd(int fd);

class TcpServer {
 public:
  TcpServer(std::string host, int port, int backlog = 128);
  ~TcpServer();

  TcpServer(const TcpServer&) = delete;
  TcpServer& operator=(const TcpServer&) = delete;

  int listen_fd() const { return listen_fd_; }
  int accept_one();
  void stop();
  bool stopping() const { return stopping_.load(); }

 private:
  std::string host_;
  int port_;
  int listen_fd_;
  std::atomic<bool> stopping_{false};
};

}  // namespace distml::net
