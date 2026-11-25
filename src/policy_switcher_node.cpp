// Several policies on one robot, switched live without a gap in control.
//
// deploy_policy_node runs one policy per process: switching means stopping one node and starting another, and in
// between XBot2 only holds the last references (a robot balancing upright falls). This node loads several policies
// and, every control period, runs the selected one exactly as deploy_policy_node does (same OnnxPolicy, command and
// sensor receivers, gains and control modes). A switch happens between two control periods; the new policy starts
// from a clean state (no previous action, empty histories, its own gains for the first second), like a freshly
// started deploy_policy_node. Joints a policy does not act on (e.g. the front legs of a 2-leg policy) are held at
// its default pose with its gains, as their PD did in training.
//
// Usage: policy_switcher_node NAME=DIR [NAME=DIR ...]   (each DIR holds policy.onnx and deploy_metadata.yaml)
// Select with std_msgs/String on /policy_switcher/select: a policy name, its number (1..N), or "release"/"0"
// (stop commanding: XBot2 holds the last references). The active policy is published on /policy_switcher/active.
// Velocity commands as for deploy_policy_node, on /policy_deploy_node/commands/<command> (the node has the same name).
// Parameters: start (policy to run right away, default none), plus deploy_policy_node's imu_name,
// obs_group_override, allow_missing_robot_joints, command_timeout_s, sensor_timeout_s, joint_velocity_limit_scale.

#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "deploy_onnx.h"
#include "ros_adaptation_receiver.h"
#include "ros_command_receiver.h"
#include "ros_sensor_receiver.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

namespace {

using namespace std::chrono_literals;
using policy_deploy::RosCommandReceiver;
using policy_deploy::RosSensorReceiver;

struct PolicyFiles {
    std::string name;
    std::string model_path;
    std::string metadata_path;
};

class PolicySwitcherNode final : public rclcpp::Node {
public:
    PolicySwitcherNode()
        // same name as deploy_policy_node: velocity commands, joystick and GUI use the same topics
        : rclcpp::Node("policy_deploy_node")
    {
        _imu_name = declare_parameter<std::string>("imu_name", "imu_link");
        _obs_group_override = declare_parameter<std::string>("obs_group_override", "");
        _allow_missing_robot_joints = declare_parameter<bool>("allow_missing_robot_joints", false);
        _command_timeout_s = declare_parameter<double>("command_timeout_s", 0.5);
        _sensor_timeout_s = declare_parameter<double>("sensor_timeout_s", 0.5);
        _start = declare_parameter<std::string>("start", "");
        if(_command_timeout_s < 0.0 || _sensor_timeout_s < 0.0)
        {
            throw std::runtime_error("Parameters 'command_timeout_s' and 'sensor_timeout_s' must be non-negative");
        }
    }

    void init(const std::vector<PolicyFiles>& policies)
    {
        // robot, as deploy_policy_node
        auto cfg = XBot::ConfigOptionsFromParams(shared_from_this(), "xbotcore/", 2s);
        _robot = XBot::RobotInterface::getRobot(cfg);
        _imu = _robot->getImu(_imu_name);
        if(!_imu)
        {
            throw std::runtime_error(std::format("IMU sensor '{}' not found in robot model", _imu_name));
        }

        _vid_to_jid.assign(_robot->getNv(), -1);
        for(int jid = 0; jid < _robot->getJointNum(); ++jid)
        {
            int vid = _robot->getVIndexFromVName(_robot->getJointNames()[jid]);
            if(vid >= 0)
            {
                _vid_to_jid[vid] = jid;
            }
        }

        _robot_info.joint_names = _robot->getVNames();
        _robot->getPose(_imu->getName(), "base_link", _robot_info.base_T_imu);
        for(const auto& [name, link] : _robot->getUrdf()->links_)
        {
            _robot_info.body_names.push_back(name);
        }
        _robot_info.body_pose_in_base = [this](const std::string& name) {
            return _robot->getPose(name, "base_link");
        };
        double joint_velocity_limit_scale = declare_parameter<double>("joint_velocity_limit_scale", 1.0);
        Eigen::VectorXd qmin, qmax;
        _robot->getJointLimits(qmin, qmax);
        _robot_info.joint_pos_min = qmin.cast<float>();
        _robot_info.joint_pos_max = qmax.cast<float>();
        Eigen::VectorXd vmax;
        _robot->getVelocityLimits(vmax);
        _robot_info.joint_vel_max = vmax.cast<float>() * joint_velocity_limit_scale;

        // load every policy once now, so a bad one fails at start-up and not at the switch
        _policies = policies;
        double control_dt = 0.0;
        for(std::size_t i = 0; i < _policies.size(); ++i)
        {
            const auto& p = _policies[i];
            auto policy = make_policy(p);
            const auto& info = policy->policyInfo();
            control_dt = (i == 0) ? info.control_dt : std::min(control_dt, info.control_dt);
            RCLCPP_INFO(get_logger(), "%zu: %s (%d actions, control period %.0f ms) from %s",
                        i + 1, p.name.c_str(), info.action_size, info.control_dt * 1000.0, p.model_path.c_str());
        }

        _adaptation_receiver = std::make_unique<policy_deploy::RosAdaptationReceiver>(*this);
        _active_publisher = create_publisher<std_msgs::msg::String>(
            "/policy_switcher/active", rclcpp::QoS(1).reliable().transient_local());
        _select_subscription = create_subscription<std_msgs::msg::String>(
            "/policy_switcher/select", 10,
            [this](std_msgs::msg::String::ConstSharedPtr msg) { on_select(msg->data); });

        _vec_nq.resize(_robot->getNq());
        _vec_nj.setZero(_robot->getJointNum());

        if(!_start.empty())
        {
            on_select(_start);
        }
        else
        {
            publish_active();
        }

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(control_dt));
        _timer = create_timer(period, [this]() { control_timer_callback(); });
    }

private:
    std::unique_ptr<XBot::policy::OnnxPolicy> make_policy(const PolicyFiles& p)
    {
        return std::make_unique<XBot::policy::OnnxPolicy>(
            p.model_path, p.metadata_path, _obs_group_override, _robot_info, _allow_missing_robot_joints);
    }

    void on_select(const std::string& choice)
    {
        int index = -1;
        if(choice == "release" || choice == "0")
        {
            if(_policy)
            {
                RCLCPP_INFO(get_logger(), "released: XBot2 holds the last references");
            }
            release();
            publish_active();
            return;
        }
        for(std::size_t i = 0; i < _policies.size(); ++i)
        {
            if(_policies[i].name == choice || std::to_string(i + 1) == choice)
            {
                index = static_cast<int>(i);
            }
        }
        if(index < 0)
        {
            RCLCPP_WARN(get_logger(), "unknown policy '%s'", choice.c_str());
            return;
        }
        if(index == _active)
        {
            return;  // already running: a repeated message must not restart it mid-motion
        }
        activate(index);
        publish_active();
    }

    void release()
    {
        _sensor_receiver.reset();
        _command_receiver.reset();
        _outputs.reset();
        _policy.reset();
        _active = -1;
    }

    void activate(int index)
    {
        // a fresh policy object, like a freshly started deploy_policy_node: no previous action, empty histories
        release();
        _policy = make_policy(_policies[index]);
        const auto& info = _policy->policyInfo();
        _outputs = std::make_unique<XBot::policy::Outputs>(info.action_size, _robot_info.joint_names.size());
        _inputs = XBot::policy::Inputs();
        _command_receiver = std::make_unique<RosCommandReceiver>(*this, *_policy, _command_timeout_s);
        _sensor_receiver = std::make_unique<RosSensorReceiver>(
            *this, *_policy, _sensor_timeout_s, _robot_info.base_T_imu.linear());
        _active = index;
        _active_since = std::chrono::steady_clock::now();
        _start_time = now();
        RCLCPP_INFO(get_logger(), "active policy: %s", _policies[index].name.c_str());
    }

    void publish_active()
    {
        std_msgs::msg::String msg;
        msg.data = _active >= 0 ? _policies[_active].name : "released";
        _active_publisher->publish(msg);
    }

    void hold_joints_not_acted_on()
    {
        // joints of the policy that it leaves uncontrolled (control mode 0) are held at its default pose with its
        // gains, with the control mode it uses for the joints it acts on
        auto& outputs = *_outputs;
        const auto& info = _policy->policyInfo();
        uint8_t mode = 0;
        for(Eigen::Index vi = 0; vi < outputs.ctrl_mode.size(); ++vi)
        {
            mode = std::max(mode, outputs.ctrl_mode(vi));
        }
        if(mode == 0)
        {
            return;
        }
        for(std::size_t pi = 0; pi < info.joint_id_policy_to_robot.size(); ++pi)
        {
            const int vi = info.joint_id_policy_to_robot[pi];
            if(vi < 0 || outputs.ctrl_mode(vi) != 0)
            {
                continue;
            }
            outputs.ctrl_mode(vi) = mode;
            outputs.q_des(vi) = info.joint_default_pos[pi];
            outputs.v_des(vi) = 0.0;
            outputs.k_des(vi) = info.stiffness[pi];
            outputs.d_des(vi) = info.damping[pi];
        }
    }

    void control_timer_callback()
    {
        if(!_policy)
        {
            return;
        }
        auto& outputs = *_outputs;

        // the rest as deploy_policy_node's control loop
        while(rclcpp::ok() && !_robot->sense())
        {
            std::this_thread::sleep_for(1ms);
        }
        if(!rclcpp::ok())
        {
            return;
        }

        _robot->positionToMinimal(_robot->getPositionReferenceFeedback(), _inputs.q_ref);
        _inputs.w_R_imu = _imu->getOrientation();
        _sensor_receiver->update_imu(_inputs.w_R_imu);

        _inputs.last_action = outputs.raw_action;
        const auto ros_now = now();
        _command_receiver->fill_inputs(_inputs, ros_now);
        if(!_sensor_receiver->fill_inputs(_inputs, ros_now))
        {
            return;
        }
        _inputs.time_sec = (now() - _start_time).seconds();
        _inputs.q = _robot->getJointPositionMinimal();
        _inputs.v = _robot->getJointVelocity();
        _inputs.tau = _robot->getJointEffort();
        _inputs.v_ref = _robot->getVelocityReferenceFeedback();
        _inputs.tau_ref = _robot->getEffortReferenceFeedback();
        _inputs.k = _robot->getStiffness();
        _inputs.d = _robot->getDamping();
        _inputs.imu_omega = _imu->getAngularVelocity();
        _inputs.imu_acc = _imu->getLinearAcceleration();

        if(_inputs.step == 0)
        {
            outputs.q_des = _inputs.q_ref;
        }
        _adaptation_receiver->fill_inputs(_inputs);
        _policy->run(_inputs, outputs);
        _inputs.step++;

        hold_joints_not_acted_on();

        for(int vi = 0; vi < _robot->getNv(); ++vi)
        {
            int jid = _vid_to_jid[vi];
            if(jid >= 0)
            {
                _vec_nj(jid) = outputs.ctrl_mode(vi);
            }
        }
        _robot->setControlMode(_vec_nj);
        _robot->minimalToPosition(outputs.q_des, _vec_nq);
        _robot->setPositionReference(_vec_nq);
        _robot->setVelocityReference(outputs.v_des);
        _robot->setEffortReference(outputs.tau_des);
        if(std::chrono::steady_clock::now() - _active_since < 1s)
        {
            // as deploy_policy_node: each policy's impedance is sent during its first second only
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
    XBot::policy::RobotInfo _robot_info;
    std::string _imu_name;
    std::string _obs_group_override;
    std::string _start;
    bool _allow_missing_robot_joints{false};
    double _command_timeout_s{0.0};
    double _sensor_timeout_s{0.0};
    std::vector<PolicyFiles> _policies;
    int _active{-1};
    std::unique_ptr<XBot::policy::OnnxPolicy> _policy;
    std::unique_ptr<RosCommandReceiver> _command_receiver;
    std::unique_ptr<RosSensorReceiver> _sensor_receiver;
    std::unique_ptr<policy_deploy::RosAdaptationReceiver> _adaptation_receiver;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _active_publisher;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _select_subscription;
    XBot::policy::Inputs _inputs;
    std::unique_ptr<XBot::policy::Outputs> _outputs;
    std::vector<int> _vid_to_jid;
    Eigen::VectorXd _vec_nq;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> _vec_nj;
    std::chrono::steady_clock::time_point _active_since;
    rclcpp::Time _start_time{0, 0, RCL_ROS_TIME};
    rclcpp::TimerBase::SharedPtr _timer;
};

} // namespace

int main(int argc, char *argv[]) {

    rclcpp::init(argc, argv);
    const auto args = rclcpp::remove_ros_arguments(argc, argv);

    std::vector<PolicyFiles> policies;
    for(std::size_t i = 1; i < args.size(); ++i)
    {
        const auto& arg = args[i];
        const auto eq = arg.find('=');
        if(eq == std::string::npos || eq == 0 || eq + 1 == arg.size())
        {
            std::cerr << "policy_switcher_node: expected NAME=DIR, got '" << arg << "'" << std::endl;
            rclcpp::shutdown();
            return 1;
        }
        const auto dir = arg.substr(eq + 1);
        policies.push_back({arg.substr(0, eq), dir + "/policy.onnx", dir + "/deploy_metadata.yaml"});
    }
    if(policies.empty())
    {
        std::cerr << "Usage: " << argv[0] << " NAME=DIR [NAME=DIR ...]  (DIR with policy.onnx and deploy_metadata.yaml)" << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    try
    {
        auto node = std::make_shared<PolicySwitcherNode>();
        node->init(policies);
        rclcpp::spin(node);
    }
    catch(const std::exception& e)
    {
        std::cerr << "policy_switcher_node: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
