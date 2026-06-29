#include "common/socket_utils.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace distml::net {

namespace {

std::string errno_message(const std::string& prefix) {
  return prefix + ": " + std::strerror(errno);
}

}  // namespace

int create_listening_socket(const std::string& host, int port, int backlog) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error(errno_message("socket"));
  }

  int reuse = 1;
  if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    close_fd(fd);
    throw std::runtime_error(errno_message("setsockopt"));
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    close_fd(fd);
    throw std::runtime_error("invalid IPv4 host: " + host);
  }

  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    close_fd(fd);
    throw std::runtime_error(errno_message("bind"));
  }
  if (::listen(fd, backlog) < 0) {
    close_fd(fd);
    throw std::runtime_error(errno_message("listen"));
  }
  return fd;
}

bool read_line(int fd, std::string* line) {
  line->clear();
  char ch = '\0';
  while (true) {
    ssize_t n = ::recv(fd, &ch, 1, 0);
    if (n == 0) {
      return !line->empty();
    }
    if (n < 0) {
      return false;
    }
    if (ch == '\n') {
      return true;
    }
    if (ch != '\r') {
      line->push_back(ch);
    }
  }
}

bool write_all(int fd, const std::string& data) {
  const char* ptr = data.data();
  std::size_t remaining = data.size();
  while (remaining > 0) {
    ssize_t n = ::send(fd, ptr, remaining, 0);
    if (n <= 0) {
      return false;
    }
    ptr += n;
    remaining -= static_cast<std::size_t>(n);
  }
  return true;
}

void close_fd(int fd) {
  if (fd >= 0) {
    ::close(fd);
  }
}

TcpServer::TcpServer(std::string host, int port, int backlog)
    : host_(std::move(host)),
      port_(port),
      listen_fd_(create_listening_socket(host_, port_, backlog)) {}

TcpServer::~TcpServer() {
  stop();
}

int TcpServer::accept_one() {
  sockaddr_in client{};
  socklen_t len = sizeof(client);
  return ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &len);
}

void TcpServer::stop() {
  bool expected = false;
  if (stopping_.compare_exchange_strong(expected, true)) {
    close_fd(listen_fd_);
  }
}

}  // namespace distml::net
