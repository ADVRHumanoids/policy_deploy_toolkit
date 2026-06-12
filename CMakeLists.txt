cmake_minimum_required(VERSION 3.20)
project(xbot2_mujoco LANGUAGES CXX)

find_package(onnxruntime REQUIRED)
find_package(xbot2 REQUIRED)

add_executable(deploy_onnx src/main.cpp)
target_link_libraries(deploy_onnx PRIVATE onnxruntime::onnxruntime xbot2::xbot2)