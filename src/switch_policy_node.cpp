#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "deploy_onnx.h"
#include "ee_command.h"
#include "ros_command_receiver.h"
#include "ros_sensor_receiver.h"
#include <rclcpp/rclcpp.hpp>
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

namespace {

using namespace std::chrono_literals;
using policy_deploy::RosCommandReceiver;
using policy_deploy::RosSensorReceiver;

// Two policies on one robot: LOCO while the base is upright, GETUP otherwise.
// Both run every step so their histories stay current; only the active one is sent to the robot.
class SwitchPolicyNode final : public rclcpp::Node {
public:
    enum Slot { LOCO = 0, GETUP = 1 };

    SwitchPolicyNode()
        : rclcpp::Node("switch_policy_node")
    {
        _node_start_time = now();
        _imu_name = declare_parameter<std::string>("imu_name", "imu_link");
        _allow_missing_robot_joints = declare_parameter<bool>("allow_missing_robot_joints", true);
        _command_timeout_s = declare_parameter<double>("command_timeout_s", 0.5);
        _sensor_timeout_s = declare_parameter<double>("sensor_timeout_s", 0.5);
        _flat_height_scan = declare_parameter<bool>("flat_heightscan", false);
        _foot_frames = declare_parameter<std::vector<std::string>>(
            "foot_frames", {"contact_1", "contact_2", "contact_3", "contact_4"});
        _foot_contact_height = declare_parameter<double>("foot_contact_height", 0.0);
        // |roll| or |pitch| above fall_tilt: GETUP; both below up_tilt for up_hold_s: LOCO
        _fall_tilt = declare_parameter<double>("fall_tilt", 0.6);
        _up_tilt = declare_parameter<double>("up_tilt", 0.3);
        _up_hold_s = declare_parameter<double>("up_hold_s", 0.5);

        if(_up_tilt > _fall_tilt)
        {
            throw std::runtime_error("Parameter 'up_tilt' must not exceed 'fall_tilt'");
        }
    }

    void init(const std::array<std::string, 2>& model_paths, const std::array<std::string, 2>& metadata_paths)
    {
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

        XBot::policy::RobotInfo robot_info;
        robot_info.joint_names = _robot->getVNames();
        _robot->getPose(_imu->getName(), "base_link", robot_info.base_T_imu);
        const double joint_velocity_limit_scale = declare_parameter<double>("joint_velocity_limit_scale", 1.0);
        Eigen::VectorXd qmin, qmax, vmax;
        _robot->getJointLimits(qmin, qmax);
        _robot->getVelocityLimits(vmax);
        robot_info.joint_pos_min = qmin.cast<float>();
        robot_info.joint_pos_max = qmax.cast<float>();
        robot_info.joint_vel_max = vmax.cast<float>() * joint_velocity_limit_scale;
        _base_R_imu = robot_info.base_T_imu.linear();

        for(int s : {LOCO, GETUP})
        {
            _policy[s] = std::make_unique<XBot::policy::OnnxPolicy>(
                model_paths[s], metadata_paths[s], "", robot_info, _allow_missing_robot_joints);
            _outputs[s] = std::make_unique<XBot::policy::Outputs>(
                _policy[s]->policyInfo().action_size, robot_info.joint_names.size());
        }

        // ponytail: ROS commands and sensors only for LOCO, the receivers declare node parameters once
        if(!_policy[GETUP]->command_specs().empty() || !_policy[GETUP]->sensor_specs().empty())
        {
            throw std::runtime_error("GETUP policy must not use commands or sensors");
        }
        const double dt = _policy[LOCO]->policyInfo().control_dt;
        if(std::abs(_policy[GETUP]->policyInfo().control_dt - dt) > 1e-9)
        {
            throw std::runtime_error("LOCO and GETUP policies must share control_dt");
        }

        _command_receiver = std::make_unique<RosCommandReceiver>(*this, *_policy[LOCO], _command_timeout_s);
        _sensor_receiver = std::make_unique<RosSensorReceiver>(*this, *_policy[LOCO], _sensor_timeout_s, _base_R_imu);

        _vec_nq.resize(_robot->getNq());
        _vec_nj.setZero(_robot->getJointNum());
        _loop_start = std::chrono::steady_clock::now();

        RCLCPP_INFO(get_logger(), "Switch: GETUP above %.2f rad tilt, LOCO below %.2f rad for %.2f s",
                    _fall_tilt, _up_tilt, _up_hold_s);

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));
        _timer = create_timer(period, [this]() { control_timer_callback(); });
    }

private:
    // largest of |roll| and |pitch| of the base (R = Rz Ry Rx)
    double tilt(const Eigen::Matrix3d& w_R_b) const
    {
        const double roll = std::atan2(w_R_b(2, 1), w_R_b(2, 2));
        const double pitch = std::asin(std::clamp(-w_R_b(2, 0), -1.0, 1.0));
        return std::max(std::abs(roll), std::abs(pitch));
    }

    void update_active(double tilt_rad, double dt)
    {
        int next = _active;
        if(_inputs.step == 0)
        {
            next = tilt_rad < _up_tilt ? LOCO : GETUP;
        }
        else if(_active == LOCO && tilt_rad > _fall_tilt)
        {
            next = GETUP;
        }
        else if(_active == GETUP)
        {
            _up_time = tilt_rad < _up_tilt ? _up_time + dt : 0.0;
            if(_up_time >= _up_hold_s)
            {
                next = LOCO;
            }
        }

        if(next != _active || _inputs.step == 0)
        {
            RCLCPP_WARN(get_logger(), "Switch: %s (tilt %.3f rad)", next == LOCO ? "LOCO" : "GETUP", tilt_rad);
            _active = next;
            _up_time = 0.0;
            _send_gains = true;
        }
    }

    void control_timer_callback()
    {
        const auto t_start = std::chrono::steady_clock::now();
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

        const Eigen::Matrix3d w_R_b = _inputs.w_R_imu.toRotationMatrix() * _base_R_imu.transpose();
        if(_flat_height_scan)
        {
            // height scan z values in the level base frame: flat ground at -h
            const Eigen::Matrix3d t_R_b = XBot::policy::remove_yaw(w_R_b);
            std::vector<Eigen::Vector3d> feet;
            Eigen::Affine3d b_T_foot;
            for(const auto& f : _foot_frames)
            {
                _robot->getPose(f, "base_link", b_T_foot);
                feet.push_back(b_T_foot.translation());
            }
            const double h = XBot::policy::pelvis_height(t_R_b, feet, _foot_contact_height);
            _inputs.height_scan = Eigen::VectorXd::Constant(_policy[LOCO]->policyInfo().height_scan_size, -h);
        }

        update_active(tilt(w_R_b), _policy[LOCO]->policyInfo().control_dt);

        if(_inputs.step == 0)
        {
            _q_sent = _inputs.q_ref;
        }
        for(int s : {LOCO, GETUP})
        {
            auto& outputs = *_outputs[s];
            // velocity clamp of the action term acts on the reference last sent to the robot
            outputs.q_des = _q_sent;
            _inputs.last_action = outputs.raw_action;
            _policy[s]->run(_inputs, outputs);
        }
        _inputs.step++;

        const auto& out = *_outputs[_active];
        _q_sent = out.q_des;
        for(int vi = 0; vi < _robot->getNv(); ++vi)
        {
            if(_vid_to_jid[vi] >= 0)
            {
                _vec_nj(_vid_to_jid[vi]) = out.ctrl_mode(vi);
            }
        }
        _robot->setControlMode(_vec_nj);
        _robot->minimalToPosition(out.q_des, _vec_nq);
        _robot->setPositionReference(_vec_nq);
        _robot->setVelocityReference(out.v_des);
        _robot->setEffortReference(out.tau_des);
        // gains for the first second and on each switch, otherwise left to external tuning
        if(t_start - _loop_start < 1s || _send_gains)
        {
            _robot->setStiffness(out.k_des);
            _robot->setDamping(out.d_des);
            _send_gains = false;
        }
#ifdef ENABLE_COMMAND_TIMESTAMPS
        _robot->setCommandTimestamp(_robot->getStateTimestamp());
#endif
        _robot->move();
    }

    XBot::RobotInterface::UniquePtr _robot;
    XBot::ImuSensor::ConstPtr _imu;
    std::string _imu_name;
    bool _allow_missing_robot_joints{true};
    double _command_timeout_s{0.0};
    double _sensor_timeout_s{0.0};
    bool _flat_height_scan{false};
    std::vector<std::string> _foot_frames;
    double _foot_contact_height{0.0};
    double _fall_tilt{0.0};
    double _up_tilt{0.0};
    double _up_hold_s{0.0};
    std::array<std::unique_ptr<XBot::policy::OnnxPolicy>, 2> _policy;
    std::array<std::unique_ptr<XBot::policy::Outputs>, 2> _outputs;
    std::unique_ptr<RosCommandReceiver> _command_receiver;
    std::unique_ptr<RosSensorReceiver> _sensor_receiver;
    XBot::policy::Inputs _inputs;
    int _active{GETUP};
    double _up_time{0.0};
    bool _send_gains{false};
    Eigen::VectorXd _q_sent;
    std::vector<int> _vid_to_jid;
    rclcpp::Time _node_start_time{0, 0, RCL_ROS_TIME};
    Eigen::VectorXd _vec_nq;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> _vec_nj;
    std::chrono::steady_clock::time_point _loop_start;
    Eigen::Matrix3d _base_R_imu{Eigen::Matrix3d::Identity()};
    rclcpp::TimerBase::SharedPtr _timer;
};

} // namespace

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    if(argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <loco_model> <loco_metadata> <getup_model> <getup_metadata>" << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    try
    {
        auto node = std::make_shared<SwitchPolicyNode>();
        node->init({argv[1], argv[3]}, {argv[2], argv[4]});
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    }
    catch(const std::exception& e)
    {
        std::cerr << "switch_policy_node: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
