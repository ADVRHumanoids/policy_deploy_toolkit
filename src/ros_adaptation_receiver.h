#pragma once

#include "types.h"

#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace policy_deploy {

class RosAdaptationReceiver {
public:
    explicit RosAdaptationReceiver(rclcpp::Node& node);

    // Copy the latest complete estimate (or initial defaults) and publish that snapshot.
    // Valid estimates are retained indefinitely.
    void fill_inputs(XBot::policy::Inputs& inputs);

private:
    using Message = std_msgs::msg::Float64MultiArray;
    rclcpp::Node& _node;
    std::mutex _mutex;
    decltype(XBot::policy::Inputs::adaptation_parameters) _value =
        XBot::policy::Inputs{}.adaptation_parameters;
    rclcpp::Subscription<Message>::SharedPtr _subscription;
    rclcpp::Publisher<Message>::SharedPtr _publisher;
};

} // namespace policy_deploy
