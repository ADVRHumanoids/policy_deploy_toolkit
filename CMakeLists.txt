cmake_minimum_required(VERSION 3.20)
project(policy_deploy_toolkit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(ament_cmake REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(xbot2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(xbot2_diagnostics QUIET)

# library
add_library(policy_deploy_toolkit STATIC
    src/deploy_onnx.cpp
    src/obs_term.cpp
    src/action_term.cpp
    src/command_term.cpp
)

target_include_directories(policy_deploy_toolkit
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(policy_deploy_toolkit
    PUBLIC
    Eigen3::Eigen
    onnxruntime::onnxruntime
    yaml-cpp::yaml-cpp
)

# node
add_executable(deploy_policy_node
    src/deploy_policy_node.cpp
    )

target_link_libraries(deploy_policy_node
    PRIVATE
    policy_deploy_toolkit
    xbot2_diagnostics::xbot2_diagnostics_ros
    xbot2::xbot2
    rclcpp::rclcpp
    std_msgs::std_msgs__rosidl_typesupport_cpp
    geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)

if(xbot2_diagnostics_FOUND)
    target_link_libraries(deploy_policy_node
        PRIVATE
        xbot2_diagnostics::xbot2_diagnostics
    )
    target_compile_definitions(deploy_policy_node
        PRIVATE
        -DHAS_XBOT2_DIAGNOSTICS
    )
endif()

# install node
install(TARGETS deploy_policy_node
    DESTINATION lib/${PROJECT_NAME}
)

# install examples
install(DIRECTORY examples/
    DESTINATION share/${PROJECT_NAME}/examples
)

# install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)   

include(CTest)

if(BUILD_TESTING)
    add_executable(command_term_test
        test/command_term_test.cpp
    )

    target_link_libraries(command_term_test
        PRIVATE
        policy_deploy_toolkit
    )

    add_test(NAME command_term_test COMMAND command_term_test)
endif()


ament_package()
