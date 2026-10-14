#include "ros_sensor_receiver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <numeric>
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
    // how each block of source pixels collapses to one policy pixel when the camera renders
    // at a multiple of the policy resolution:
    //   center: pixel at the block centre, same single ray per pixel the sim renders
    //   mean / median: smooth sensor noise; min: keeps the closest obstacle in the block
    // mean/min/median skip invalid (non-finite) pixels; an all-invalid block reads as far
    _depth_downsample = _node.declare_parameter<std::string>("depth_downsample", "center");
    if(_depth_downsample != "center" && _depth_downsample != "mean" &&
       _depth_downsample != "min" && _depth_downsample != "median")
    {
        throw std::runtime_error("Parameter 'depth_downsample' must be one of center, mean, min, median; got '" +
                                 _depth_downsample + "'");
    }

    for(const auto& spec : policy.sensor_specs())
    {
        if(std::holds_alternative<XBot::policy::HeightScanSpec>(spec))
        {
            create_height_scan_subscription(std::get<XBot::policy::HeightScanSpec>(spec));
        }
        else if(std::holds_alternative<XBot::policy::DepthSpec>(spec))
        {
            create_depth_subscription(std::get<XBot::policy::DepthSpec>(spec));
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
    if(_height_scan_states.empty() && _depth_states.empty())
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    for(auto& [name, state] : _depth_states)
    {
        if(!state.has_value)
        {
            RCLCPP_WARN_THROTTLE(_node.get_logger(),
                                 *_node.get_clock(),
                                 1000,
                                 "Waiting for depth image '%s'",
                                 name.c_str());
            return false;
        }

        if((now - state.stamp).seconds() > _timeout_s)
        {
            RCLCPP_WARN_THROTTLE(_node.get_logger(),
                                 *_node.get_clock(),
                                 1000,
                                 "Depth image '%s' is stale; reusing last valid frame",
                                 name.c_str());
        }

        inputs.depth[state.spec.obs_group] = state.value;
    }

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

void RosSensorReceiver::create_depth_subscription(const XBot::policy::DepthSpec& spec)
{
    if(spec.size() <= 0)
    {
        throw std::runtime_error("Depth sensor '" + spec.name + "' has non-positive size");
    }

    DepthState state;
    state.spec = spec;
    // 1.0 = empty, so an unfilled buffer reads as "nothing ahead" rather than a wall
    state.value.assign(spec.size(), 1.0f);
    state.stamp = _node.now();
    state.input_pub = _node.create_publisher<sensor_msgs::msg::Image>(
        "~/sensors/" + spec.name + "/policy_input", rclcpp::SensorDataQoS());
    _depth_states.emplace(spec.name, std::move(state));

    // own group, so a multi-threaded executor runs image processing beside the control timer
    // (default group) instead of delaying its tick; store_depth_image publishes under _mutex
    if(!_depth_group)
    {
        _depth_group = _node.create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    }
    rclcpp::SubscriptionOptions options;
    options.callback_group = _depth_group;

    using Message = sensor_msgs::msg::Image;
    const auto topic = "~/sensors/" + spec.name + "/depth";
    auto subscription = _node.create_subscription<Message>(
        topic,
        rclcpp::SensorDataQoS().keep_last(1).best_effort(),
        [this, name = spec.name](Message::ConstSharedPtr msg) {
            store_depth_image(name, *msg);
        },
        options);

    _subscriptions.push_back(std::move(subscription));
    RCLCPP_INFO(_node.get_logger(), "Subscribed to depth '%s' on '%s' (%dx%d)",
                spec.name.c_str(), topic.c_str(), spec.width, spec.height);
}

void RosSensorReceiver::store_depth_image(const std::string& name, const sensor_msgs::msg::Image& msg)
{
    auto it = _depth_states.find(name);
    if(it == _depth_states.end())
    {
        return;
    }

    const auto& spec = it->second.spec;

    // The camera may render larger than the policy resolution (gz is unreliable at 48x30), as
    // long as it divides evenly; each output pixel then takes one source sample.
    if(static_cast<int>(msg.width) % spec.width != 0 ||
       static_cast<int>(msg.height) % spec.height != 0)
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                             "Ignoring depth '%s': %ux%u is not an integer multiple of %dx%d",
                             name.c_str(), msg.width, msg.height, spec.width, spec.height);
        return;
    }

    if(msg.encoding != "32FC1")
    {
        RCLCPP_WARN_THROTTLE(_node.get_logger(), *_node.get_clock(), 1000,
                             "Ignoring depth '%s': expected 32FC1, got '%s'",
                             name.c_str(), msg.encoding.c_str());
        return;
    }

    const double span = spec.far - spec.near;
    if(span <= 0.0)
    {
        return;
    }

    // pooling a full-res frame takes milliseconds: build outside the lock the control loop reads under
    std::vector<float> out(spec.size());

    const int row_step = static_cast<int>(msg.height) / spec.height;
    const int col_step = static_cast<int>(msg.width) / spec.width;
    const float near = static_cast<float>(spec.near);
    const float far = static_cast<float>(spec.far);
    const auto pixel = [&](int r, int c) {
        return reinterpret_cast<const float*>(msg.data.data() + r * msg.step)[c];
    };

    std::vector<float> block;
    block.reserve(row_step * col_step);

    for(int row = 0; row < spec.height; ++row)
    {
        for(int col = 0; col < spec.width; ++col)
        {
            const int r0 = row * row_step;
            const int c0 = col * col_step;
            float d = far;

            if(_depth_downsample == "center")
            {
                // centre of the source block, so the sampled ray matches where the pixel centre points;
                // misses (nan/inf) read as empty, matching nan_to_num(nan=far, ...) in training
                d = pixel(r0 + row_step / 2, c0 + col_step / 2);
                d = std::isfinite(d) ? std::clamp(d, near, far) : far;
            }
            else
            {
                block.clear();
                for(int r = r0; r < r0 + row_step; ++r)
                {
                    for(int c = c0; c < c0 + col_step; ++c)
                    {
                        const float v = pixel(r, c);
                        if(std::isfinite(v))
                        {
                            // clamp first so returns beyond far don't drag the mean past the window
                            block.push_back(std::clamp(v, near, far));
                        }
                    }
                }

                if(block.empty())
                {
                    d = far;
                }
                else if(_depth_downsample == "mean")
                {
                    d = std::accumulate(block.begin(), block.end(), 0.0f) / block.size();
                }
                else if(_depth_downsample == "min")
                {
                    d = *std::min_element(block.begin(), block.end());
                }
                else  // median
                {
                    // ponytail: upper median for even counts, no averaging of the two middles
                    auto mid = block.begin() + block.size() / 2;
                    std::nth_element(block.begin(), mid, block.end());
                    d = *mid;
                }
            }

            out[row * spec.width + col] = (d - near) / static_cast<float>(span);
        }
    }

    // exactly the buffer the ONNX image input receives: 32FC1, [0, 1] over [near, far], 1 = empty
    if(it->second.input_pub->get_subscription_count() > 0)
    {
        sensor_msgs::msg::Image img;
        img.header = msg.header;
        img.height = spec.height;
        img.width = spec.width;
        img.encoding = "32FC1";
        img.is_bigendian = false;
        img.step = sizeof(float) * spec.width;
        img.data.resize(sizeof(float) * out.size());
        std::memcpy(img.data.data(), out.data(), img.data.size());
        it->second.input_pub->publish(std::move(img));
    }

    std::lock_guard<std::mutex> lock(_mutex);
    it->second.value = std::move(out);
    it->second.stamp = _node.now();
    it->second.has_value = true;
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
