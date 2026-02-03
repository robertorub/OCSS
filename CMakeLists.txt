cmake_minimum_required(VERSION 3.10)
project(OptimalControlSolver)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 CONFIG REQUIRED)

# include_directories(
#     ${CMAKE_SOURCE_DIR}/src/include
#     /opt/homebrew/opt/eigen/include/eigen3
# )

add_executable(test_suite src/examples/test_suite.cpp)
target_link_libraries(test_suite PUBLIC Eigen3::Eigen)
target_include_directories(test_suite PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
