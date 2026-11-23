#pragma once

#include "types.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace policy_deploy {

// Publish slices of the existing model vectors without processing terms a second time.
class RosPolicyTermPublisher {
public:
    RosPolicyTermPublisher(rclcpp::Node& node, const std::string& prefix,
                           const std::vector<XBot::policy::PolicyTermSpec>& specs)
        : _specs(specs)
    {
        for(const auto& spec : _specs)
        {
            _publishers.push_back(node.create_publisher<Message>(prefix + "/" + spec.name, 10));
        }
    }

    template<typename Derived>
    void publish(const Eigen::MatrixBase<Derived>& values)
    {
        for(std::size_t i = 0; i < _specs.size(); ++i)
        {
            const auto& spec = _specs[i];
            Message message;
            message.data.resize(spec.size);
            for(int j = 0; j < spec.size; ++j)
            {
                message.data[j] = static_cast<float>(values[spec.offset + j]);
            }
            _publishers[i]->publish(message);
        }
    }

private:
    using Message = std_msgs::msg::Float32MultiArray;
    std::vector<XBot::policy::PolicyTermSpec> _specs;
    std::vector<rclcpp::Publisher<Message>::SharedPtr> _publishers;
};

} // namespace policy_deploy
