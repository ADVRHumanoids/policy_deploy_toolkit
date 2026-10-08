#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "deploy_onnx.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace policy_deploy {

class RosSensorReceiver {
public:
    RosSensorReceiver(rclcpp::Node& node,
                      const XBot::policy::OnnxPolicy& policy,
                      double timeout_s,
                      const Eigen::Matrix3d& base_R_imu);

    // Latest orientation maps IMU coordinates into world coordinates.
    void update_imu(const Eigen::Quaterniond& orientation);

    bool fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now);

private:
    friend struct RosSensorReceiverTest;
    RosSensorReceiver(rclcpp::Node& node, double timeout_s, const Eigen::Matrix3d& base_R_imu);

    struct PendingScan {
        std::vector<Eigen::Vector3d> points;
        std::string frame;
        rclcpp::Time received;
    };
    struct HeightScanState {
        XBot::policy::HeightScanSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
        std::optional<PendingScan> pending;
    };

    void create_height_scan_subscription(const XBot::policy::HeightScanSpec& spec);

    void store_height_scan(const std::string& name, const sensor_msgs::msg::PointCloud2& msg);

    static const sensor_msgs::msg::PointField* find_field(const sensor_msgs::msg::PointCloud2& msg,
                                                        const std::string& name);

    static std::size_t scalar_size(const sensor_msgs::msg::PointField& field);

    bool process_pending(HeightScanState& state, const rclcpp::Time& now);

    static double read_scalar(const std::uint8_t* data, const sensor_msgs::msg::PointField& field);

    rclcpp::Node& _node;
    double _timeout_s;
    Eigen::Matrix3d _base_R_imu;
    tf2_ros::Buffer _tf_buffer;
    tf2_ros::TransformListener _tf_listener;
    std::map<std::string, Eigen::Affine3d> _base_T_source;
    std::optional<Eigen::Quaterniond> _latest_imu;
    std::mutex _mutex;
    std::map<std::string, HeightScanState> _height_scan_states;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> _subscriptions;
};

} // namespace policy_deploy
