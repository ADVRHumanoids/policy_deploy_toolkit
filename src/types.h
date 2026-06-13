#ifndef __XBOT2_DEPLOY_POLICY_TYPES_H
#define __XBOT2_DEPLOY_POLICY_TYPES_H

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace XBot::policy {

    struct RobotInfo {
    std::vector<std::string> joint_names;
    Eigen::Affine3d base_T_imu;
};

struct PolicyInfo {
    double control_dt;
    std::vector<std::string> joint_names;
    std::vector<double> stiffness, damping;
    std::vector<double> joint_default_pos;
    std::vector<int> joint_id_policy_to_robot;
    std::vector<int> joint_id_robot_to_policy;
    std::map<std::string, int> command_size;
    int action_size;
};

struct Inputs {
    std::map<std::string, Eigen::VectorXd> command; 
    Eigen::VectorXd last_action;

    Eigen::VectorXd q; 
    Eigen::VectorXd v; 
    Eigen::VectorXd tau;

    Eigen::VectorXd q_ref;
    Eigen::VectorXd v_ref;
    Eigen::VectorXd tau_ref;
    Eigen::VectorXd k;
    Eigen::VectorXd d;

    Eigen::Quaterniond w_R_imu;
    Eigen::Vector3d imu_omega;
    Eigen::Vector3d imu_acc;
};

struct Outputs {
    Eigen::VectorXd raw_action;

    Eigen::VectorXd q_des;
    Eigen::VectorXd v_des;
    Eigen::VectorXd tau_des;
    Eigen::VectorXd k_des;
    Eigen::VectorXd d_des;

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> ctrl_mode;

    Outputs(int action_size, int num_joints) {
        raw_action = Eigen::VectorXd::Zero(action_size);
        q_des = Eigen::VectorXd::Zero(num_joints);
        v_des = Eigen::VectorXd::Zero(num_joints);
        tau_des = Eigen::VectorXd::Zero(num_joints);
        k_des = Eigen::VectorXd::Zero(num_joints);
        d_des = Eigen::VectorXd::Zero(num_joints);
        ctrl_mode = decltype(ctrl_mode)::Zero(num_joints);
    }
};

}

#endif // __XBOT2_DEPLOY_POLICY_TYPES_H