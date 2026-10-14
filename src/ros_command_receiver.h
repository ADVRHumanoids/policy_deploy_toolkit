#pragma once

#include <map>
#include <string>
#include <vector>

#include "deploy_onnx.h"
#include <rclcpp/rclcpp.hpp>

namespace policy_deploy {

class RosCommandReceiver {
public:
    RosCommandReceiver(rclcpp::Node& node,
                       const XBot::policy::OnnxPolicy& policy,
                       double timeout_s);

    void fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now);

private:
    struct CommandState {
        XBot::policy::CommandSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };

    XBot::policy::Inputs _inputs;

    static bool is_velocity_command(const XBot::policy::CommandSpec& spec);

    void create_velocity_subscription(const std::string& name);

    void create_raw_subscription(const std::string& name);

    void store_command(const std::string& name, const Eigen::VectorXd& raw);

    rclcpp::Node& _node;
    const XBot::policy::OnnxPolicy& _policy;
    double _timeout_s;
    std::map<std::string, Eigen::VectorXd> _defaults;
    std::map<std::string, CommandState> _states;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> _subscriptions;
};

} // namespace policy_deploy
