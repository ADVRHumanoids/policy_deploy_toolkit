#include <chrono>

#include "deploy_onnx.h"
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

int main(int argc, char *argv[]) {

    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("xbot2_deploy_policy");

    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <model_metadata_path>" << std::endl;
        return 1;
    }

    // build robotinterface
    auto cfg = XBot::ConfigOptionsFromParams(node, "xbotcore/", 2s);
    auto robot = XBot::RobotInterface::getRobot(cfg);
    auto imu = robot->getImu("imu_link");

    // fill robot info for policy
    XBot::policy::RobotInfo robot_info;
    robot_info.joint_names = robot->getVNames();
    robot->getPose(imu->getName(), "base_link", robot_info.base_T_imu);

    // build policy wrapper
    std::string model_path = argv[1];
    std::string model_metadata_path = argv[2];
    XBot::policy::OnnxPolicy policy(model_path, model_metadata_path, robot_info);
    
    // construct policy wrapper inputs and outputs
    XBot::policy::Inputs inputs;
    XBot::policy::Outputs outputs(policy.policyInfo().action_size, robot_info.joint_names.size());

    // buffers
    Eigen::VectorXd vec_nq(robot->getNq());

    auto loop_start = std::chrono::steady_clock::now();

    while(rclcpp::ok())
    {
        auto t_start = std::chrono::steady_clock::now();

        // wait for robot to be updated
        while(!robot->sense())
        {
            std::this_thread::sleep_for(1ms);
        }
        
        // fill inputs for policy
        inputs.last_action = outputs.raw_action; // for the first iteration, last action is zero
        inputs.command["base_velocity"] = Eigen::Vector3d(0.1, 0.2, 0.3);
        inputs.q = robot->getJointPositionMinimal();
        inputs.v = robot->getJointVelocity();
        inputs.tau = robot->getJointEffort();
        robot->positionToMinimal(robot->getPositionReferenceFeedback(), inputs.q_ref);
        inputs.v_ref = robot->getVelocityReferenceFeedback();
        inputs.tau_ref = robot->getEffortReferenceFeedback();
        inputs.k = robot->getStiffness();
        inputs.d = robot->getDamping();
        inputs.w_R_imu = imu->getOrientation();
        inputs.imu_omega = imu->getAngularVelocity();
        inputs.imu_acc = imu->getLinearAcceleration();

        // run policy
        policy.run(inputs, outputs);

        // send commands to robot
        robot->setControlMode(outputs.ctrl_mode);
        robot->minimalToPosition(outputs.q_des, vec_nq);
        robot->setPositionReference(vec_nq);
        robot->setVelocityReference(outputs.v_des);
        robot->setEffortReference(outputs.tau_des);
        if(t_start - loop_start < 1s)
        {
            // note: we stop sending impedance after one second,
            // so we can tune it online from a separate node (e.g. gui)
            robot->setStiffness(outputs.k_des);
            robot->setDamping(outputs.d_des);
        }
        robot->setCommandTimestamp(robot->getStateTimestamp());        
        robot->move();

        // wait for next control cycle
        std::this_thread::sleep_until(t_start + std::chrono::duration<double>(policy.policyInfo().control_dt));
    }

}