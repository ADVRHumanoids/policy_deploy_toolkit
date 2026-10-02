#include "ros_command_receiver.h"

#include <cstddef>
#include <utility>

#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace policy_deploy {

RosCommandReceiver::RosCommandReceiver(rclcpp::Node& node,
                                       const XBot::policy::OnnxPolicy& policy,
                                       double timeout_s):
    _node(node),
    _policy(policy),
    _timeout_s(timeout_s),
    _defaults(policy.default_commands())
{
    for(const auto& spec : policy.command_specs())
    {
        CommandState state;
        state.spec = spec;
        state.value = _defaults.at(spec.name);
        state.stamp = _node.now();
        _states.emplace(spec.name, std::move(state));

        if(is_velocity_command(spec))
        {
            create_velocity_subscription(spec.name);
        }
        else
        {
            create_raw_subscription(spec.name);
        }
    }
}

void RosCommandReceiver::fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now)
{
    for(auto& [name, state] : _states)
    {
        const bool stale = !state.has_value || (now - state.stamp).seconds() > _timeout_s;
        inputs.command[name] = stale ? _defaults.at(name) : state.value;
    }
    _inputs = inputs;
}

bool RosCommandReceiver::is_velocity_command(const XBot::policy::CommandSpec& spec)
{
    return spec.class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:VelocityCommand" ||
           spec.class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:TerrainBasedVelocityCommandPLAY" ||
           spec.class_type == "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand";
}

void RosCommandReceiver::create_velocity_subscription(const std::string& name)
{
    using Message = geometry_msgs::msg::Twist;

    const auto topic = "~/commands/" + name;
    auto subscription = _node.create_subscription<Message>(
        topic,
        rclcpp::QoS(1).best_effort(),
        [this, name](Message::ConstSharedPtr msg) {
            Eigen::VectorXd raw(3);
            raw << msg->linear.x, msg->linear.y, msg->angular.z;
            store_command(name, raw);
        });

    _subscriptions.push_back(std::move(subscription));
    RCLCPP_INFO(_node.get_logger(), "Subscribed to command '%s' on '%s'", name.c_str(), topic.c_str());
}

void RosCommandReceiver::create_raw_subscription(const std::string& name)
{
    using Message = std_msgs::msg::Float64MultiArray;

    const auto topic = "~/commands/" + name + "/raw";
    auto subscription = _node.create_subscription<Message>(
        topic,
        rclcpp::QoS(1).best_effort(),
        [this, name](Message::ConstSharedPtr msg) {
            Eigen::VectorXd raw(static_cast<int>(msg->data.size()));
            for(int i = 0; i < raw.size(); ++i)
            {
                raw(i) = msg->data[static_cast<std::size_t>(i)];
            }

            store_command(name, raw);
        });

    _subscriptions.push_back(std::move(subscription));
    RCLCPP_INFO(_node.get_logger(), "Subscribed to command '%s' on '%s'", name.c_str(), topic.c_str());
}

void RosCommandReceiver::store_command(const std::string& name, const Eigen::VectorXd& raw)
{
    Eigen::VectorXd sanitized;
    std::string reason;
    if(!_policy.sanitize_command(_inputs,name, raw, sanitized, &reason))
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring command '%s': %s",
                             name.c_str(),
                             reason.c_str());
        return;
    }

    auto it = _states.find(name);
    if(it == _states.end())
    {
        RCLCPP_WARN(_node.get_logger(), "Ignoring unknown command '%s'", name.c_str());
        return;
    }

    it->second.value = std::move(sanitized);
    it->second.stamp = _node.now();
    it->second.has_value = true;
}

} // namespace policy_deploy
