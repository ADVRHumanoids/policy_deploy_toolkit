#include "ros_adaptation_receiver.h"

#include <algorithm>
#include <cmath>

namespace policy_deploy {

RosAdaptationReceiver::RosAdaptationReceiver(rclcpp::Node& node): _node(node)
{
    _publisher = _node.create_publisher<Message>("~/adaptation_parameters/current", 1);
    _subscription = _node.create_subscription<Message>(
        "~/adaptation_parameters", rclcpp::QoS(1).best_effort(),
        [this](Message::ConstSharedPtr msg) {
            if(msg->data.size() != 8 ||
               !std::all_of(msg->data.begin(), msg->data.end(),
                            [](double value) { return std::isfinite(value); }) ||
               std::any_of(msg->data.begin() + 4, msg->data.end(),
                           [](double value) { return value < 0.; }))
            {
                RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                    "Ignoring adaptation estimate: expected eight finite values with nonnegative friction");
                return;
            }
            std::lock_guard<std::mutex> lock(_mutex);
            _value = Eigen::Map<const Eigen::Matrix<double, 8, 1>>(msg->data.data());
            std::cout << "Received adaptation parameters: " << _value.transpose() << std::endl;
        });
}

void RosAdaptationReceiver::fill_inputs(XBot::policy::Inputs& inputs)
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        inputs.adaptation_parameters = _value;
    }
    Message msg;
    msg.data.assign(inputs.adaptation_parameters.data(),
                    inputs.adaptation_parameters.data() + inputs.adaptation_parameters.size());
    _publisher->publish(msg);
}

} // namespace policy_deploy
