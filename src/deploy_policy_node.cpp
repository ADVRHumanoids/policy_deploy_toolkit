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
#include "ros_command_receiver.h"
#include "ros_sensor_receiver.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

namespace {

using namespace std::chrono_literals;
using policy_deploy::RosCommandReceiver;
using policy_deploy::RosSensorReceiver;

class PolicyDeployNode final : public rclcpp::Node {
public:
    PolicyDeployNode()
        : rclcpp::Node("policy_deploy_node")
    {
        _node_start_time = now();
        _imu_name = declare_parameter<std::string>("imu_name", "imu_link");
        _obs_group_override = declare_parameter<std::string>("obs_group_override", "");
        _allow_missing_robot_joints = declare_parameter<bool>("allow_missing_robot_joints", false);
        _command_timeout_s = declare_parameter<double>("command_timeout_s", 0.5);
        _sensor_timeout_s = declare_parameter<double>("sensor_timeout_s", 0.5);

        RCLCPP_INFO(get_logger(), "Obs group override: %s", _obs_group_override.c_str());
        RCLCPP_INFO(get_logger(), "Allow missing robot joints: %s", _allow_missing_robot_joints ? "true" : "false");

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

        double joint_velocity_limit_scale = declare_parameter<double>("joint_velocity_limit_scale", 1.0);
        RCLCPP_INFO(get_logger(), "Joint velocity limit scale: %f", joint_velocity_limit_scale);

        Eigen::VectorXd qmin, qmax;
        _robot->getJointLimits(qmin, qmax);
        robot_info.joint_pos_min = qmin.cast<float>();
        robot_info.joint_pos_max = qmax.cast<float>();
        Eigen::VectorXd vmax;
        _robot->getVelocityLimits(vmax);
        robot_info.joint_vel_max = vmax.cast<float>() * joint_velocity_limit_scale;

        // build policy wrapper
        _policy = std::make_unique<XBot::policy::OnnxPolicy>(model_path, model_metadata_path, _obs_group_override, robot_info, _allow_missing_robot_joints);

        // construct policy wrapper outputs
        _outputs = std::make_unique<XBot::policy::Outputs>(
            _policy->policyInfo().action_size,
            robot_info.joint_names.size());

        _command_receiver = std::make_unique<RosCommandReceiver>(*this, *_policy, _command_timeout_s);
        _sensor_receiver = std::make_unique<RosSensorReceiver>(*this, *_policy, _sensor_timeout_s, robot_info.base_T_imu.linear());
        _observation_publisher = create_publisher<std_msgs::msg::Float32MultiArray>("policy/observations", 10);
        _action_publisher = create_publisher<std_msgs::msg::Float32MultiArray>("policy/actions", 10);
        const auto& recurrent_state_names = _policy->recurrent_state_names();
        for(std::size_t i = 0; i < recurrent_state_names.size(); ++i)
        {
            const auto topic = "policy/hidden_states/state_" + std::to_string(i);
            _hidden_state_publishers.push_back(
                create_publisher<std_msgs::msg::Float32MultiArray>(topic, 10));
            RCLCPP_INFO(get_logger(), "Publishing recurrent state '%s' on '%s'",
                        recurrent_state_names[i].c_str(), topic.c_str());
        }

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

        _inputs.w_R_imu = _imu->getOrientation();
        _sensor_receiver->update_imu(_inputs.w_R_imu);

        // fill inputs for policy
        _inputs.last_action = outputs.raw_action; // for the first iteration, last action is zero
        const auto ros_now = now();
        _command_receiver->fill_inputs(_inputs, ros_now);
        if(!_sensor_receiver->fill_inputs(_inputs, ros_now))
        {
            return;
        }
        _inputs.time_sec = (now() - _node_start_time).seconds();
        _inputs.q = _robot->getJointPositionMinimal();
        _inputs.v = _robot->getJointVelocity();
        _inputs.tau = _robot->getJointEffort();
        _robot->positionToMinimal(_robot->getPositionReferenceFeedback(), _inputs.q_ref);
        _inputs.v_ref = _robot->getVelocityReferenceFeedback();
        _inputs.tau_ref = _robot->getEffortReferenceFeedback();
        _inputs.k = _robot->getStiffness();
        _inputs.d = _robot->getDamping();
        _inputs.imu_omega = _imu->getAngularVelocity();
        _inputs.imu_acc = _imu->getLinearAcceleration();

        // initialize outputs as needed
        if(_inputs.step == 0)
        {
            outputs.q_des = _inputs.q_ref;
        }

        // run policy
        _policy->run(_inputs, outputs);

        // increment step counter
        _inputs.step++;

        std_msgs::msg::Float32MultiArray observation_msg;
            observation_msg.data.assign(
                outputs.raw_observation.data(),
                outputs.raw_observation.data() + outputs.raw_observation.size());
        _observation_publisher->publish(observation_msg);

        std_msgs::msg::Float32MultiArray action_msg;
        action_msg.data.resize(outputs.raw_action.size());
        for(Eigen::Index i = 0; i < outputs.raw_action.size(); ++i)
        {
            action_msg.data[i] = static_cast<float>(outputs.raw_action[i]);
        }
        _action_publisher->publish(action_msg);

        const auto& recurrent_state_names = _policy->recurrent_state_names();
        for(std::size_t i = 0; i < outputs.raw_recurrent_states.size(); ++i)
        {
            std_msgs::msg::Float32MultiArray state_msg;
                const auto& recurrent_state = outputs.raw_recurrent_states[i];
                state_msg.data.assign(recurrent_state.data(), recurrent_state.data() + recurrent_state.size());
            std_msgs::msg::MultiArrayDimension state_dimension;
            state_dimension.label = recurrent_state_names[i];
            state_dimension.size = static_cast<uint32_t>(state_msg.data.size());
            state_dimension.stride = state_dimension.size;
            state_msg.layout.dim.push_back(state_dimension);
            _hidden_state_publishers[i]->publish(state_msg);
        }

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
    bool _allow_missing_robot_joints{false};
    double _command_timeout_s{0.0};
    double _sensor_timeout_s{0.0};
    std::unique_ptr<XBot::policy::OnnxPolicy> _policy;
    std::unique_ptr<RosCommandReceiver> _command_receiver;
    std::unique_ptr<RosSensorReceiver> _sensor_receiver;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr _observation_publisher;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr _action_publisher;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr> _hidden_state_publishers;
    XBot::policy::Inputs _inputs;
    std::unique_ptr<XBot::policy::Outputs> _outputs;
    std::vector<int> _vid_to_jid;
    rclcpp::Time _node_start_time{0, 0, RCL_ROS_TIME};
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
