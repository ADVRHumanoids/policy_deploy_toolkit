cmake_minimum_required(VERSION 3.20)
project(xbot2_mujoco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(onnxruntime REQUIRED)
find_package(xbot2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(deploy_onnx 
    src/main.cpp 
    src/deploy_onnx.cpp 
    src/obs_term.cpp
    src/action_term.cpp
    )

target_link_libraries(deploy_onnx 
    PRIVATE 
    onnxruntime::onnxruntime 
    xbot2::xbot2 
    rclcpp::rclcpp 
    std_msgs::std_msgs__rosidl_typesupport_cpp
)

install(TARGETS deploy_onnx
    DESTINATION lib/${PROJECT_NAME}
)