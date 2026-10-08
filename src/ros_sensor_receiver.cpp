#include "ros_sensor_receiver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xbot2_diagnostics/ros2_publisher.h>

namespace policy_deploy {

RosSensorReceiver::RosSensorReceiver(rclcpp::Node& node, double timeout_s,
                                     const Eigen::Matrix3d& base_R_imu):
    _node(node),
    _timeout_s(timeout_s),
    _base_R_imu(base_R_imu),
    _tf_buffer(node.get_clock()),
    _tf_listener(_tf_buffer, &node, false)
{
}

RosSensorReceiver::RosSensorReceiver(rclcpp::Node& node,
                                     const XBot::policy::OnnxPolicy& policy,
                                     double timeout_s,
                                     const Eigen::Matrix3d& base_R_imu):
    RosSensorReceiver(node, timeout_s, base_R_imu)
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

void RosSensorReceiver::update_imu(const Eigen::Quaterniond& orientation)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const double norm = orientation.norm();
    if(!orientation.coeffs().allFinite() || !std::isfinite(norm) || norm < 1e-12)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                            "Ignoring invalid IMU orientation");
        return;
    }
    _latest_imu = orientation.normalized();
}

bool RosSensorReceiver::process_pending(HeightScanState& state, const rclcpp::Time& now)
{
    if(!state.pending)
    {
        return false;
    }
    auto& pending = *state.pending;
    if((now - pending.received).seconds() > _timeout_s)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                            "Discarding height scan '%s': IMU/TF wait expired", state.spec.name.c_str());
        state.pending.reset();
        return false;
    }
    if(!_latest_imu)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                            "Waiting for IMU orientation for height scan '%s'", state.spec.name.c_str());
        if(_timeout_s == 0.0) state.pending.reset();
        return false;
    }
    auto transform = _base_T_source.find(pending.frame);
    if(transform == _base_T_source.end())
    {
        Eigen::Affine3d base_T_source = Eigen::Affine3d::Identity();
        if(pending.frame != "base_link")
        {
            try
            {
                // Source-to-base extrinsics are static: lookup latest once, never wait.
                const auto tf = _tf_buffer.lookupTransform("base_link", pending.frame, tf2::TimePointZero);
                const auto& q = tf.transform.rotation;
                const auto& t = tf.transform.translation;
                Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
                if(!rotation.coeffs().allFinite() || rotation.norm() < 1e-12 ||
                   !std::isfinite(rotation.norm()))
                {
                    throw std::runtime_error("invalid TF rotation");
                }
                base_T_source.linear() = rotation.normalized().toRotationMatrix();
                base_T_source.translation() = Eigen::Vector3d(t.x, t.y, t.z);
                if(!base_T_source.translation().allFinite())
                {
                    throw std::runtime_error("invalid TF translation");
                }
            }
            catch(const std::exception& error)
            {
                RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                                    "Waiting for static TF base_link <- %s: %s",
                                    pending.frame.c_str(), error.what());
                if(_timeout_s == 0.0) state.pending.reset();
                return false;
            }
        }
        transform = _base_T_source.emplace(pending.frame, base_T_source).first;
    }
    const Eigen::Matrix3d world_R_base = _latest_imu->toRotationMatrix() * _base_R_imu.transpose();
    if(world_R_base.col(0).head<2>().norm() < 1e-12)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                            "Ignoring height scan '%s': base yaw is undefined", state.spec.name.c_str());
        state.pending.reset();
        return false;
    }
    const double yaw = std::atan2(world_R_base(1, 0), world_R_base(0, 0));
    const Eigen::Matrix3d horizontal_R_base =
        Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * world_R_base;
    Eigen::VectorXd scan(pending.points.size());
    for(std::size_t i = 0; i < pending.points.size(); ++i)
    {
        const Eigen::Vector3d point = horizontal_R_base * (transform->second * pending.points[i]);
        if(!point.allFinite())
        {
            RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                                "Ignoring height scan '%s': transformed point is not finite", state.spec.name.c_str());
            state.pending.reset();
            return false;
        }
        scan(i) = point.z();
    }
    state.value = std::move(scan);
    state.stamp = pending.received;
    state.has_value = true;
    state.pending.reset();
    return true;
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
        process_pending(state, now);
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
            if(msg->header.stamp.sec >= 0)
            {
                const auto stamp = rclcpp::Time(msg->header.stamp, _node.get_clock()->get_clock_type());
                stats_pub->update_and_publish((_node.now() - stamp).seconds());
            }
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

    std::lock_guard<std::mutex> lock(_mutex);
    it->second.pending.reset();
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

    auto reject = [&](const char* reason) {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                            "Ignoring height scan '%s': %s", name.c_str(), reason);
    };
    const auto now = _node.now();
    if(msg.header.frame_id.empty())
    {
        reject("empty frame ID");
        return;
    }
    if(msg.is_bigendian)
    {
        reject("big-endian PointCloud2 data is unsupported");
        return;
    }
    const sensor_msgs::msg::PointField* fields[] = {
        find_field(msg, "x"), find_field(msg, "y"), find_field(msg, "z")};
    std::size_t last_field_end = 0;
    for(const auto* field : fields)
    {
        if(!field || field->count < 1 ||
           (field->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
            field->datatype != sensor_msgs::msg::PointField::FLOAT64))
        {
            reject("XYZ fields must be float32 or float64");
            return;
        }
        last_field_end = std::max(last_field_end,
            static_cast<std::size_t>(field->offset) + scalar_size(*field));
    }
    if(msg.point_step == 0 ||
       msg.row_step < static_cast<std::size_t>(msg.width) * msg.point_step ||
       last_field_end > msg.point_step)
    {
        reject("malformed PointCloud2 layout");
        return;
    }
    const auto min_data_size = (point_count == 0) ? 0 :
        (static_cast<std::size_t>(msg.height) - 1) * msg.row_step +
        (static_cast<std::size_t>(msg.width) - 1) * msg.point_step + last_field_end;
    if(msg.data.size() < min_data_size)
    {
        reject("PointCloud2 data is shorter than its layout");
        return;
    }
    PendingScan pending{{}, msg.header.frame_id, now};
    pending.points.reserve(point_count);
    for(std::size_t row = 0; row < msg.height; ++row)
    {
        for(std::size_t col = 0; col < msg.width; ++col)
        {
            const auto offset = row * msg.row_step + col * msg.point_step;
            Eigen::Vector3d point;
            for(int axis = 0; axis < 3; ++axis)
            {
                point(axis) = read_scalar(msg.data.data() + offset + fields[axis]->offset, *fields[axis]);
            }
            if(!point.allFinite())
            {
                reject("XYZ value is not finite");
                return;
            }
            pending.points.push_back(point);
        }
    }
    it->second.pending = std::move(pending);
    // Consume immediately when IMU and static TF are available.
    process_pending(it->second, now);
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

double RosSensorReceiver::read_scalar(const std::uint8_t* data, const sensor_msgs::msg::PointField& field)
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
