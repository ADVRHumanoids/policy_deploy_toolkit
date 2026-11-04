#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "deploy_onnx.h"
#include "ee_command.h"
#include "ros_command_receiver.h"
#include "ros_sensor_receiver.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <std_msgs/msg/string.hpp>
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
        // flat ground: every height scan point at the ground below the pelvis, no scan topic
        _flat_height_scan = declare_parameter<bool>("flat_heightscan", false);
        // pelvis height above flat ground: lowest foot frame, standing foot_contact_height above the ground
        _foot_frames = declare_parameter<std::vector<std::string>>(
            "foot_frames", {"contact_1", "contact_2", "contact_3", "contact_4"});
        _foot_contact_height = declare_parameter<double>("foot_contact_height", 0.0);

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

        _base_R_imu = robot_info.base_T_imu.linear();
        for(const auto& spec : _policy->command_specs())
        {
            if(spec.class_type.ends_with(":EETrajectoryCommand"))
            {
                init_ee_command(spec.name);
            }
        }

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

        // simulated power loss: comma-separated "joint[:scale]" (scale = kept gain fraction, default 0),
        // empty string clears the fault
        _fault_sub = create_subscription<std_msgs::msg::String>(
            "~/fault_joints", rclcpp::QoS(1).reliable(),
            [this](const std_msgs::msg::String::SharedPtr msg) { fault_callback(msg->data); });

        _timer = create_timer(period, [this]() { control_timer_callback(); });
    }

private:
    // reference: per arm (left, right) target pos (3), target quat wxyz (4), goal pos (3), goal quat wxyz (4)
    void init_ee_command(const std::string& name)
    {
        _ee_command_name = name;
        _ee_frames = declare_parameter<std::vector<std::string>>("ee_frames", {"wrist_yaw_1_link", "wrist_yaw_2_link"});
        // task frame broadcast as TF <ee_task_parent_frame> -> ee_task, for visualization
        _ee_task_parent_frame = declare_parameter<std::string>("ee_task_parent_frame", "base_link");
        _tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        if(_ee_frames.size() != 2)
        {
            throw std::runtime_error("Parameter 'ee_frames' must list the left and right EE frames");
        }

        _ee_reference_sub = create_subscription<std_msgs::msg::Float64MultiArray>(
            "~/commands/" + name + "/reference", rclcpp::QoS(1).best_effort(),
            [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
                if(msg->data.size() != 28)
                {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                         "Ignoring EE reference with %zu values (expected 28)", msg->data.size());
                    return;
                }
                const auto pose = [&msg](std::size_t i) {
                    const double* d = msg->data.data() + i;
                    XBot::policy::EePose p;
                    p.pos << d[0], d[1], d[2];
                    p.quat = Eigen::Quaterniond(d[3], d[4], d[5], d[6]).normalized();
                    return p;
                };
                for(std::size_t arm = 0; arm < 2; ++arm)
                {
                    _ee_target[arm] = pose(14 * arm);
                    _ee_goal[arm] = pose(14 * arm + 7);
                }
                _ee_has_reference = true;
            });
        _ee_measured_pub = create_publisher<std_msgs::msg::Float64MultiArray>("~/ee_measured", 10);

        RCLCPP_INFO(get_logger(), "EE command '%s': reference on '%s', EE frames %s/%s, foot contact height %.3f m",
                    name.c_str(), _ee_reference_sub->get_topic_name(),
                    _ee_frames[0].c_str(), _ee_frames[1].c_str(), _foot_contact_height);
    }

    // base orientation with yaw removed (from the IMU) and pelvis height above flat ground (from the feet)
    std::pair<Eigen::Matrix3d, double> level_frame_and_height()
    {
        const Eigen::Matrix3d t_R_b = XBot::policy::remove_yaw(_inputs.w_R_imu.toRotationMatrix() * _base_R_imu.transpose());

        std::vector<Eigen::Vector3d> feet;
        Eigen::Affine3d b_T_foot;
        for(const auto& f : _foot_frames)
        {
            _robot->getPose(f, "base_link", b_T_foot);
            feet.push_back(b_T_foot.translation());
        }
        return {t_R_b, XBot::policy::pelvis_height(t_R_b, feet, _foot_contact_height)};
    }

    // measured EE poses in the task frame, then the 26-dim command; publishes [left pose, right pose, h]
    void update_ee_command(const Eigen::Matrix3d& t_R_b, double h)
    {
        Eigen::Affine3d b_T_x;
        std::array<XBot::policy::EePose, 2> meas;
        for(std::size_t arm = 0; arm < 2; ++arm)
        {
            _robot->getPose(_ee_frames[arm], "base_link", b_T_x);
            meas[arm] = XBot::policy::to_task_frame(t_R_b, h, b_T_x);
        }

        // without a reference, hold the EE poses measured at the first step
        if(!_ee_has_reference)
        {
            _ee_target = meas;
            _ee_goal = meas;
            _ee_has_reference = true;
        }

        Eigen::VectorXd command(26);
        std_msgs::msg::Float64MultiArray msg;
        for(std::size_t arm = 0; arm < 2; ++arm)
        {
            command.segment<13>(13 * arm) = XBot::policy::ee_command(
                meas[arm], _ee_target[arm], _ee_goal[arm], _policy->policyInfo().control_dt);
            const auto& q = meas[arm].quat;
            msg.data.insert(msg.data.end(), {meas[arm].pos.x(), meas[arm].pos.y(), meas[arm].pos.z(),
                                             q.w(), q.x(), q.y(), q.z()});
        }
        msg.data.push_back(h);
        _inputs.command[_ee_command_name] = command;
        _ee_measured_pub->publish(msg);

        // task frame in the base frame: yaw-only axes, origin on the ground under the pelvis
        const Eigen::Matrix3d b_R_t = t_R_b.transpose();
        const Eigen::Vector3d b_p_t = b_R_t * Eigen::Vector3d(0.0, 0.0, -h);
        const Eigen::Quaterniond b_q_t(b_R_t);
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = now();
        tf.header.frame_id = _ee_task_parent_frame;
        tf.child_frame_id = "ee_task";
        tf.transform.translation.x = b_p_t.x();
        tf.transform.translation.y = b_p_t.y();
        tf.transform.translation.z = b_p_t.z();
        tf.transform.rotation.w = b_q_t.w();
        tf.transform.rotation.x = b_q_t.x();
        tf.transform.rotation.y = b_q_t.y();
        tf.transform.rotation.z = b_q_t.z();
        _tf_broadcaster->sendTransform(tf);

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "EE: pelvis height %.3f m, left (%.3f %.3f %.3f), right (%.3f %.3f %.3f)",
                             h, meas[0].pos.x(), meas[0].pos.y(), meas[0].pos.z(), meas[1].pos.x(), meas[1].pos.y(), meas[1].pos.z());
    }

    void fault_callback(const std::string& data)
    {
        _faults.clear();
        std::stringstream ss(data);
        std::string item;
        while(std::getline(ss, item, ','))
        {
            std::erase(item, ' ');
            if(item.empty())
            {
                continue;
            }
            const auto sep = item.find(':');
            const std::string name = item.substr(0, sep);
            double scale = 0.0;
            if(sep != std::string::npos)
            {
                try
                {
                    scale = std::clamp(std::stod(item.substr(sep + 1)), 0.0, 1.0);
                }
                catch(const std::exception&)
                {
                    RCLCPP_WARN(get_logger(), "fault: bad scale in '%s', ignored", item.c_str());
                    continue;
                }
            }
            int vid = _robot->getVIndexFromVName(name);
            if(vid < 0)
            {
                RCLCPP_WARN(get_logger(), "fault: unknown joint '%s', ignored", name.c_str());
                continue;
            }
            _faults.emplace_back(vid, scale);
        }
        _fault_changed = true;
        RCLCPP_WARN(get_logger(), "fault: power loss on %zu joint(s) [%s]", _faults.size(), data.c_str());
    }

    void apply_fault(XBot::policy::Outputs& outputs)
    {
        for(const auto& [vid, scale] : _faults)
        {
            outputs.k_des(vid) *= scale;
            outputs.d_des(vid) *= scale;
            outputs.tau_des(vid) *= scale;
            outputs.ctrl_mode(vid) |= (8 + 16); // stiffness + damping
        }
    }

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

        if(!_ee_command_name.empty() || _flat_height_scan)
        {
            const auto [t_R_b, h] = level_frame_and_height();
            if(!_ee_command_name.empty())
            {
                update_ee_command(t_R_b, h);
            }
            // height scan z values in the level base frame: flat ground at -h
            if(_flat_height_scan)
            {
                _inputs.height_scan = Eigen::VectorXd::Constant(_policy->policyInfo().height_scan_size, -h);
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Flat height scan: pelvis height %.3f m", h);
            }
        }

        // initialize outputs as needed
        if(_inputs.step == 0)
        {
            outputs.q_des = _inputs.q_ref;
        }

        // run policy
        _policy->run(_inputs, outputs);
        apply_fault(outputs);

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
        if(t_start - _loop_start < 1s || !_faults.empty() || _fault_changed)
        {
            // note: we stop sending impedance after one second,
            // so we can tune it online from a separate node (e.g. gui)
            // while a fault is active (and once after clearing) nominal gains are resent
            _robot->setStiffness(outputs.k_des);
            _robot->setDamping(outputs.d_des);
            _fault_changed = false;
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
    bool _flat_height_scan{false};
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
    std::vector<std::pair<int, double>> _faults; // (v index, kept gain fraction)
    bool _fault_changed{false};
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _fault_sub;
    Eigen::Matrix3d _base_R_imu{Eigen::Matrix3d::Identity()};
    std::string _ee_command_name;
    std::vector<std::string> _ee_frames;
    std::vector<std::string> _foot_frames;
    double _foot_contact_height{0.0};
    std::string _ee_task_parent_frame;
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;
    std::array<XBot::policy::EePose, 2> _ee_target;
    std::array<XBot::policy::EePose, 2> _ee_goal;
    bool _ee_has_reference{false};
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _ee_reference_sub;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr _ee_measured_pub;
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
