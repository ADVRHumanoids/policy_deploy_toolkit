#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "deploy_onnx.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace policy_deploy {

class RosSensorReceiver {
public:
    RosSensorReceiver(rclcpp::Node& node,
                      const XBot::policy::OnnxPolicy& policy,
                      double timeout_s);

    bool fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now);

private:
    struct HeightScanState {
        XBot::policy::HeightScanSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };

    void create_height_scan_subscription(const XBot::policy::HeightScanSpec& spec);

    void store_height_scan(const std::string& name, const sensor_msgs::msg::PointCloud2& msg);

    static const sensor_msgs::msg::PointField* find_field(const sensor_msgs::msg::PointCloud2& msg,
                                                        const std::string& name);

    static std::size_t scalar_size(const sensor_msgs::msg::PointField& field);

    static double read_z(const std::uint8_t* data, const sensor_msgs::msg::PointField& field);

    rclcpp::Node& _node;
    double _timeout_s;
    std::mutex _mutex;
    std::map<std::string, HeightScanState> _height_scan_states;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> _subscriptions;
};

} // namespace policy_deploy
