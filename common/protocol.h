#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace distml::protocol {

std::vector<std::string> split_ws(const std::string& line);
bool parse_int(const std::string& token, int* out);
bool parse_i64(const std::string& token, std::int64_t* out);
bool parse_double(const std::string& token, double* out);
std::string format_double(double value);
std::string join_doubles(const std::vector<double>& values);

template <typename T>
std::string to_string_precise(T value) {
  std::ostringstream out;
  out.precision(17);
  out << value;
  return out.str();
}

}  // namespace distml::protocol
