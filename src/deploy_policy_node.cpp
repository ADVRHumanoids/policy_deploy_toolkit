#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "deploy_onnx.h"
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

#include <xbot2_diagnostics/ros2_publisher.h>

namespace {

using namespace std::chrono_literals;

class RosCommandReceiver {
public:
    RosCommandReceiver(rclcpp::Node& node,
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

    void fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now)
    {
        for(auto& [name, state] : _states)
        {
            const bool stale = !state.has_value || (now - state.stamp).seconds() > _timeout_s;
            inputs.command[name] = stale ? _defaults.at(name) : state.value;
        }
        _inputs = inputs;
    }

private:
    struct CommandState {
        XBot::policy::CommandSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };
    
    XBot::policy::Inputs _inputs;

    static bool is_velocity_command(const XBot::policy::CommandSpec& spec)
    {
        return spec.class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:VelocityCommand" ||
               spec.class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:TerrainBasedVelocityCommandPLAY";
    }

    void create_velocity_subscription(const std::string& name)
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

    void create_raw_subscription(const std::string& name)
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

    void store_command(const std::string& name, const Eigen::VectorXd& raw)
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

    rclcpp::Node& _node;
    const XBot::policy::OnnxPolicy& _policy;
    double _timeout_s;
    std::map<std::string, Eigen::VectorXd> _defaults;
    std::map<std::string, CommandState> _states;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> _subscriptions;
};

class RosSensorReceiver {
public:
    RosSensorReceiver(rclcpp::Node& node,
                      const XBot::policy::OnnxPolicy& policy,
                      double timeout_s):
        _node(node),
        _timeout_s(timeout_s)
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

    bool fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now)
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

private:
    struct HeightScanState {
        XBot::policy::HeightScanSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };

    struct DepthState {
        XBot::policy::DepthSpec spec;
        std::vector<float> value;  // normalized to [0, 1], row-major
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr input_pub;  // echoes value for debugging
        rclcpp::Time stamp;
        bool has_value{false};
    };

    void create_depth_subscription(const XBot::policy::DepthSpec& spec)
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

    void store_depth_image(const std::string& name, const sensor_msgs::msg::Image& msg)
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

    void create_height_scan_subscription(const XBot::policy::HeightScanSpec& spec)
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

    void store_height_scan(const std::string& name, const sensor_msgs::msg::PointCloud2& msg)
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

    static const sensor_msgs::msg::PointField* find_field(const sensor_msgs::msg::PointCloud2& msg,
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

    static std::size_t scalar_size(const sensor_msgs::msg::PointField& field)
    {
        return field.datatype == sensor_msgs::msg::PointField::FLOAT64 ? sizeof(double) : sizeof(float);
    }

    static double read_z(const std::uint8_t* data, const sensor_msgs::msg::PointField& field)
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

    rclcpp::Node& _node;
    double _timeout_s;
    std::string _depth_downsample;
    rclcpp::CallbackGroup::SharedPtr _depth_group;
    std::mutex _mutex;
    std::map<std::string, HeightScanState> _height_scan_states;
    std::map<std::string, DepthState> _depth_states;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> _subscriptions;
};

class PolicyDeployNode final : public rclcpp::Node {
public:
    PolicyDeployNode()
        : rclcpp::Node("policy_deploy_node")
    {
        _imu_name = declare_parameter<std::string>("imu_name", "imu_link");
        _obs_group_override = declare_parameter<std::string>("obs_group_override", "");
        _command_timeout_s = declare_parameter<double>("command_timeout_s", 0.5);
        _sensor_timeout_s = declare_parameter<double>("sensor_timeout_s", 0.5);

        if(_command_timeout_s < 0.0)
        {
            throw std::runtime_error("Parameter 'command_timeout_s' must be non-negative");
        }

        if(_sensor_timeout_s < 0.0)
        {
            throw std::runtime_error("Parameter 'sensor_timeout_s' must be non-negative");
        }
    }

    void init(const std::string& model_path, const std::string& model_metadata_path)
    {
        // build robotinterface
        auto cfg = XBot::ConfigOptionsFromParams(shared_from_this(), "xbotcore/", 2s);
        _robot = XBot::RobotInterface::getRobot(cfg);
        _imu = _robot->getImu(_imu_name);
        if(!_imu)
        {
            throw std::runtime_error(
                std::format("IMU sensor '{}' not found in robot model", _imu_name));
        }

        // v index to joint index (needed for control mode mapping)
        // note: for simplicity, the policy only deals with v indices, whereas
        // control modes are indexed by joint indices
        _vid_to_jid.assign(_robot->getNv(), -1);
        for(int jid = 0; jid < _robot->getJointNum(); ++jid)    {
            int vid = _robot->getVIndexFromVName(_robot->getJointNames()[jid]);
            if(vid >= 0)
            {
                _vid_to_jid[vid] = jid;
            }
        }

        // fill robot info for policy
        XBot::policy::RobotInfo robot_info;
        robot_info.joint_names = _robot->getVNames();
        _robot->getPose(_imu->getName(), "base_link", robot_info.base_T_imu);

        // build policy wrapper
        _policy = std::make_unique<XBot::policy::OnnxPolicy>(model_path, model_metadata_path, _obs_group_override, robot_info);

        // construct policy wrapper outputs
        _outputs = std::make_unique<XBot::policy::Outputs>(
            _policy->policyInfo().action_size,
            robot_info.joint_names.size());

        _command_receiver = std::make_unique<RosCommandReceiver>(*this, *_policy, _command_timeout_s);
        _sensor_receiver = std::make_unique<RosSensorReceiver>(*this, *_policy, _sensor_timeout_s);

        // buffers
        _vec_nq.resize(_robot->getNq());
        _vec_nj.setZero(_robot->getJointNum());

        _loop_start = std::chrono::steady_clock::now();
        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(_policy->policyInfo().control_dt));

        _timer = create_timer(period, [this]() { control_timer_callback(); });
    }

private:
    void fill_ctrl_mode()
    {
        for(int vi = 0; vi < _robot->getNv(); ++vi)
        {
            int jid = _vid_to_jid[vi];
            if(jid >= 0)
            {
                _vec_nj(jid) = _outputs->ctrl_mode(vi);
            }
        }
    }

    void control_timer_callback()
    {
        auto& outputs = *_outputs;
        auto t_start = std::chrono::steady_clock::now();

        // wait for robot to be updated
        while(rclcpp::ok() && !_robot->sense())
        {
            std::this_thread::sleep_for(1ms);
        }
        if(!rclcpp::ok())
        {
            return;
        }

        // fill inputs for policy
        _inputs.last_action = outputs.raw_action; // for the first iteration, last action is zero
        const auto ros_now = now();
        _command_receiver->fill_inputs(_inputs, ros_now);
        if(!_sensor_receiver->fill_inputs(_inputs, ros_now))
        {
            return;
        }
        _inputs.q = _robot->getJointPositionMinimal();
        _inputs.v = _robot->getJointVelocity();
        _inputs.tau = _robot->getJointEffort();
        _robot->positionToMinimal(_robot->getPositionReferenceFeedback(), _inputs.q_ref);
        _inputs.v_ref = _robot->getVelocityReferenceFeedback();
        _inputs.tau_ref = _robot->getEffortReferenceFeedback();
        _inputs.k = _robot->getStiffness();
        _inputs.d = _robot->getDamping();
        _inputs.w_R_imu = _imu->getOrientation();
        _inputs.imu_omega = _imu->getAngularVelocity();
        _inputs.imu_acc = _imu->getLinearAcceleration();

        // run policy
        _policy->run(_inputs, outputs);

        //
        fill_ctrl_mode();

        // send commands to robot
        _robot->setControlMode(_vec_nj);
        _robot->minimalToPosition(outputs.q_des, _vec_nq);
        _robot->setPositionReference(_vec_nq);
        _robot->setVelocityReference(outputs.v_des);
        _robot->setEffortReference(outputs.tau_des);
        if(t_start - _loop_start < 1s)
        {
            // note: we stop sending impedance after one second,
            // so we can tune it online from a separate node (e.g. gui)
            _robot->setStiffness(outputs.k_des);
            _robot->setDamping(outputs.d_des);
        }
#ifdef ENABLE_COMMAND_TIMESTAMPS
        _robot->setCommandTimestamp(_robot->getStateTimestamp());
#endif
        _robot->move();
    }

    XBot::RobotInterface::UniquePtr _robot;
    XBot::ImuSensor::ConstPtr _imu;
    std::string _imu_name;
    std::string _obs_group_override;
    double _command_timeout_s{0.0};
    double _sensor_timeout_s{0.0};
    std::unique_ptr<XBot::policy::OnnxPolicy> _policy;
    std::unique_ptr<RosCommandReceiver> _command_receiver;
    std::unique_ptr<RosSensorReceiver> _sensor_receiver;
    XBot::policy::Inputs _inputs;
    std::unique_ptr<XBot::policy::Outputs> _outputs;
    std::vector<int> _vid_to_jid;
    Eigen::VectorXd _vec_nq;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> _vec_nj;
    std::chrono::steady_clock::time_point _loop_start;
    rclcpp::TimerBase::SharedPtr _timer;
};

} // namespace

int main(int argc, char *argv[]) {

    rclcpp::init(argc, argv);

    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <model_metadata_path>" << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    try
    {
        auto node = std::make_shared<PolicyDeployNode>();
        node->init(argv[1], argv[2]);
        // 2 threads: the default group (control timer, commands, height scan) stays serialized on
        // one, depth images are processed on the other
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    }
    catch(const std::exception& e)
    {
        std::cerr << "xbot2_deploy_policy: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;

}
