cmake_minimum_required(VERSION 3.20)
project(dist_ml LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(distml_common
  common/protocol.cpp
  common/socket_utils.cpp
)
target_include_directories(distml_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(distml_common PUBLIC Threads::Threads)
target_compile_options(distml_common PRIVATE -Wall -Wextra -Wpedantic)

add_library(param_server_core
  param_server/aggregator.cpp
  param_server/ps_server.cpp
)
target_include_directories(param_server_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(param_server_core PUBLIC distml_common)
target_compile_options(param_server_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(param_server
  param_server/main.cpp
)
target_link_libraries(param_server PRIVATE param_server_core)

add_library(serving_core
  serving/autoscaler.cpp
  serving/batcher.cpp
  serving/inference_server.cpp
)
target_include_directories(serving_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(serving_core PUBLIC distml_common)
target_compile_options(serving_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(inference_server
  serving/main.cpp
)
target_link_libraries(inference_server PRIVATE serving_core)

enable_testing()
add_executable(cpp_core_test
  test/cpp_core_test.cpp
)
target_link_libraries(cpp_core_test PRIVATE param_server_core serving_core)
add_test(NAME cpp_core_test COMMAND cpp_core_test)
