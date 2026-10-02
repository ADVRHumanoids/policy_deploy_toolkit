#include "ros_sensor_receiver.h"

#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xbot2_diagnostics/ros2_publisher.h>

namespace policy_deploy {

RosSensorReceiver::RosSensorReceiver(rclcpp::Node& node,
                                     const XBot::policy::OnnxPolicy& policy,
                                     double timeout_s):
    _node(node),
    _timeout_s(timeout_s)
{
    for(const auto& spec : policy.sensor_specs())
    {
        if(std::holds_alternative<XBot::policy::HeightScanSpec>(spec))
        {
            create_height_scan_subscription(std::get<XBot::policy::HeightScanSpec>(spec));
        }
        else
        {
            throw std::runtime_error("Unsupported sensor spec in ROS receiver");
        }
    }
}

bool RosSensorReceiver::fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now)
{
    if(_height_scan_states.empty())
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    for(auto& [name, state] : _height_scan_states)
    {
        if(!state.has_value)
        {
            RCLCPP_WARN_THROTTLE(_node.get_logger(),
                                 *_node.get_clock(),
                                 1000,
                                 "Waiting for height scan '%s'",
                                 name.c_str());
            return false;
        }

        if((now - state.stamp).seconds() > _timeout_s)
        {
            RCLCPP_WARN_THROTTLE(_node.get_logger(),
                                 *_node.get_clock(),
                                 1000,
                                 "Height scan '%s' is stale; reusing last valid scan",
                                 name.c_str());
        }

        inputs.height_scan = state.value;
    }

    return true;
}

void RosSensorReceiver::create_height_scan_subscription(const XBot::policy::HeightScanSpec& spec)
{
    if(spec.size <= 0)
    {
        throw std::runtime_error("Height scan sensor '" + spec.name + "' has non-positive size");
    }

    HeightScanState state;
    state.spec = spec;
    state.value = Eigen::VectorXd::Zero(spec.size);
    state.stamp = _node.now();
    _height_scan_states.emplace(spec.name, std::move(state));

    auto stats_pub = std::make_shared<XBot::diagnostics::Ros2StatsPublisher>(
        _node,
        std::format("/controller/{}/{}/delay", _node.get_name(), spec.name),
        "deploy_policy_node",
        "delay",
        1.0
    );

    using Message = sensor_msgs::msg::PointCloud2;
    const auto topic = "~/sensors/" + spec.name + "/points";
    auto subscription = _node.create_subscription<Message>(
        topic,
        rclcpp::SensorDataQoS().keep_last(1).best_effort(),
        [this, name = spec.name, stats_pub](Message::ConstSharedPtr msg) {
            const auto delay = (_node.now() - rclcpp::Time(msg->header.stamp)).seconds();
            stats_pub->update_and_publish(delay);
            store_height_scan(name, *msg);
        });

    _subscriptions.push_back(std::move(subscription));
    RCLCPP_INFO(_node.get_logger(), "Subscribed to height scan '%s' on '%s'", spec.name.c_str(), topic.c_str());
}

void RosSensorReceiver::store_height_scan(const std::string& name, const sensor_msgs::msg::PointCloud2& msg)
{
    auto it = _height_scan_states.find(name);
    if(it == _height_scan_states.end())
    {
        RCLCPP_WARN(_node.get_logger(), "Ignoring unknown height scan '%s'", name.c_str());
        return;
    }

    const auto expected_size = static_cast<std::size_t>(it->second.spec.size);
    const auto point_count = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
    if(point_count != expected_size)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': expected %zu points, got %zu",
                             name.c_str(),
                             expected_size,
                             point_count);
        return;
    }

    const auto* z_field = find_field(msg, "z");
    if(!z_field)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': PointCloud2 has no z field",
                             name.c_str());
        return;
    }

    if(msg.is_bigendian)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': big-endian PointCloud2 data is unsupported",
                             name.c_str());
        return;
    }

    if(z_field->count < 1 ||
       (z_field->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
        z_field->datatype != sensor_msgs::msg::PointField::FLOAT64))
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': z field must be float32 or float64",
                             name.c_str());
        return;
    }

    if(msg.point_step == 0 ||
       msg.row_step < static_cast<std::size_t>(msg.width) * msg.point_step ||
       z_field->offset + scalar_size(*z_field) > msg.point_step)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': malformed PointCloud2 layout",
                             name.c_str());
        return;
    }

    const auto min_data_size = (point_count == 0) ? 0 :
        (static_cast<std::size_t>(msg.height) - 1) * msg.row_step +
        (static_cast<std::size_t>(msg.width) - 1) * msg.point_step +
        z_field->offset + scalar_size(*z_field);
    if(msg.data.size() < min_data_size)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(),
                             *_node.get_clock(),
                             1000,
                             "Ignoring height scan '%s': PointCloud2 data is shorter than its layout",
                             name.c_str());
        return;
    }

    Eigen::VectorXd scan(static_cast<int>(point_count));
    for(std::size_t row = 0; row < msg.height; ++row)
    {
        for(std::size_t col = 0; col < msg.width; ++col)
        {
            const auto i = row * static_cast<std::size_t>(msg.width) + col;
            const auto offset = row * msg.row_step + col * msg.point_step + z_field->offset;
            scan(static_cast<int>(i)) = read_z(msg.data.data() + offset, *z_field);
            if(!std::isfinite(scan(static_cast<int>(i))))
            {
                RCLCPP_WARN_THROTTLE(_node.get_logger(),
                                     *_node.get_clock(),
                                     1000,
                                     "Ignoring height scan '%s': z value %zu is not finite",
                                     name.c_str(),
                                     i);
                return;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        it->second.value = std::move(scan);
        it->second.stamp = _node.now();
        it->second.has_value = true;
    }
}

const sensor_msgs::msg::PointField* RosSensorReceiver::find_field(const sensor_msgs::msg::PointCloud2& msg,
                                                               const std::string& name)
{
    for(const auto& field : msg.fields)
    {
        if(field.name == name)
        {
            return &field;
        }
    }
    return nullptr;
}

std::size_t RosSensorReceiver::scalar_size(const sensor_msgs::msg::PointField& field)
{
    return field.datatype == sensor_msgs::msg::PointField::FLOAT64 ? sizeof(double) : sizeof(float);
}

double RosSensorReceiver::read_z(const std::uint8_t* data, const sensor_msgs::msg::PointField& field)
{
    if(field.datatype == sensor_msgs::msg::PointField::FLOAT64)
    {
        double value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    float value;
    std::memcpy(&value, data, sizeof(value));
    return static_cast<double>(value);
}

} // namespace policy_deploy
