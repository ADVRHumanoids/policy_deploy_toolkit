#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
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
    }

private:
    struct CommandState {
        XBot::policy::CommandSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };

    static bool is_velocity_command(const XBot::policy::CommandSpec& spec)
    {
        return spec.class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:VelocityCommand";
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
        if(!_policy.sanitize_command(name, raw, sanitized, &reason))
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

    bool fill_inputs(XBot::policy::Inputs& inputs, const rclcpp::Time& now)
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

private:
    struct HeightScanState {
        XBot::policy::HeightScanSpec spec;
        Eigen::VectorXd value;
        rclcpp::Time stamp;
        bool has_value{false};
    };

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
    std::mutex _mutex;
    std::map<std::string, HeightScanState> _height_scan_states;
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
        _robot->setCommandTimestamp(_robot->getStateTimestamp());
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
        rclcpp::spin(node);
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
