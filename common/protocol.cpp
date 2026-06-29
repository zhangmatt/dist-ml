#include "common/protocol.h"

#include <charconv>
#include <cstdlib>
#include <iomanip>

namespace distml::protocol {

std::vector<std::string> split_ws(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

bool parse_int(const std::string& token, int* out) {
  const char* begin = token.data();
  const char* end = token.data() + token.size();
  auto [ptr, ec] = std::from_chars(begin, end, *out);
  return ec == std::errc{} && ptr == end;
}

bool parse_i64(const std::string& token, std::int64_t* out) {
  const char* begin = token.data();
  const char* end = token.data() + token.size();
  auto [ptr, ec] = std::from_chars(begin, end, *out);
  return ec == std::errc{} && ptr == end;
}

bool parse_double(const std::string& token, double* out) {
  char* parsed_end = nullptr;
  *out = std::strtod(token.c_str(), &parsed_end);
  return parsed_end != token.c_str() && parsed_end != nullptr && *parsed_end == '\0';
}

std::string format_double(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string join_doubles(const std::vector<double>& values) {
  std::ostringstream out;
  out << std::setprecision(17);
  for (double value : values) {
    out << ' ' << value;
  }
  return out.str();
}

}  // namespace distml::protocol
