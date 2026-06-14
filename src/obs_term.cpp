#include "obs_term.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace XBot::policy {

std::unique_ptr<ObsTerm> ObsTerm::create(std::string func, RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config)
{
    if(func == "isaaclab.envs.mdp.observations:base_ang_vel") 
    {
        return std::make_unique<IsaacLabBaseAngVelObsTerm>(robot_info, policy_info, config);
    } 
    else if(func == "isaaclab.envs.mdp.observations:projected_gravity")
    {
        return std::make_unique<IsaacLabProjectedGravityObsTerm>(robot_info, policy_info, config);
    }
    else if(func == "isaaclab.envs.mdp.observations:generated_commands")
    {
        return std::make_unique<IsaacLabGeneratedCommandsObsTerm>(robot_info, policy_info, config);
    }
    else if(func == "isaaclab.envs.mdp.observations:joint_pos_rel")
    {
        return std::make_unique<IsaacLabJointPosRelObsTerm>(robot_info, policy_info, config);
    }
    else if(func == "kyon_isaac.tasks.locomotion.velocity.mdp.observations:joint_pos_error")
    {
        return std::make_unique<KyonIsaacJointPosErrorObsTerm>(robot_info, policy_info, config);
    }
    else if(func == "isaaclab.envs.mdp.observations:joint_vel_rel")
    {
        return std::make_unique<KyonIsaacJointVelRelObsTerm>(robot_info, policy_info, config);
    }
    else if(func == "isaaclab.envs.mdp.observations:last_action")
    {
        return std::make_unique<IsaacLabLastActionObsTerm>(robot_info, policy_info, config);
    }
    else 
    {
        throw std::runtime_error("Unsupported observation function: " + func);
    }
}

ObsTerm::~ObsTerm() = default;

ObsTerm::ObsTerm(RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config):
        _robot_info(std::move(robot_info)),
        _policy_info(std::move(policy_info))
{
    // sanity
    if(auto n = config["clip"]; n.as<std::string>() != "null")
    {
        throw std::runtime_error("Clipping not supported for ObsTerms yet");
    }

    if(auto n = config["scale"]; n.as<std::string>() != "null")
    {
        throw std::runtime_error("Scaling not supported for ObsTerms yet");
    }

    // history
    if(auto n = config["history_length"])
    {
        _n_history = n.as<int>();
        _n_history = std::max(1, _n_history);
    }
    
    // policy joint ids
    try
    {
        _joint_ids = config["params"]["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    }
    catch(...)
    {

    }
}

int ObsTerm::size() const 
{ 
    if(_size < 0)
    {
        throw std::runtime_error("ObsTerm size not set");
    }

    return _size * _n_history; 
}

void ObsTerm::process(const Inputs& inputs, Eigen::VectorXd& output)
{
    Eigen::VectorXd term_output;
    process_impl(inputs, term_output);

    // TODO: Apply scaling and clipping if needed (not implemented yet)

    if(_n_history == 1)
    {
        output = term_output;
        return;
    }

    // Initialize histrory buffer if not already
    if(_history_buffer.size() == 0)
    {
        _history_buffer.resize(size());
        _history_buffer = term_output.replicate(_n_history, 1);
    }

    // Shift history and append new output
    _history_buffer.head((_n_history - 1) * term_output.size()) = _history_buffer.tail((_n_history - 1) * term_output.size());
    _history_buffer.tail(term_output.size()) = term_output;

    // Return the full history buffer as output
    output = _history_buffer;
}

IsaacLabBaseAngVelObsTerm::IsaacLabBaseAngVelObsTerm(RobotInfo robot_info, 
                                                     PolicyInfo policy_info,
                                                     YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = 3;
}

void IsaacLabBaseAngVelObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    // Extract the base angular velocity from the IMU readings
    output = _robot_info.base_T_imu.linear() * inputs.imu_omega;
}

IsaacLabProjectedGravityObsTerm::IsaacLabProjectedGravityObsTerm(RobotInfo robot_info, 
                                                                 PolicyInfo policy_info,
                                                                 YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = 3;
}

void IsaacLabProjectedGravityObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    // World z -> IMU -> base
    output = -_robot_info.base_T_imu.linear() * inputs.w_R_imu.toRotationMatrix().transpose().col(2);
}

IsaacLabGeneratedCommandsObsTerm::IsaacLabGeneratedCommandsObsTerm(RobotInfo robot_info,
                                                                   PolicyInfo policy_info,
                                                                   YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _command_name = config["params"]["command_name"].as<std::string>();
    _size = policy_info.command_size.at(_command_name);
}

void IsaacLabGeneratedCommandsObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    output = inputs.command.at(_command_name);
}

IsaacLabJointPosRelObsTerm::IsaacLabJointPosRelObsTerm(RobotInfo robot_info,
                                                       PolicyInfo policy_info,
                                                       YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = _joint_ids.size();
}

void IsaacLabJointPosRelObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    output.resize(_joint_ids.size());
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        output(i) = inputs.q(robot_id) - _policy_info.joint_default_pos[policy_id];
    }
}

KyonIsaacJointPosErrorObsTerm::KyonIsaacJointPosErrorObsTerm(RobotInfo robot_info,
                                                            PolicyInfo policy_info,
                                                            YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = _joint_ids.size();
}

void KyonIsaacJointPosErrorObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    output.resize(_joint_ids.size());
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        output(i) = inputs.q(robot_id) - inputs.q_ref(robot_id);
    }
}

KyonIsaacJointVelRelObsTerm::KyonIsaacJointVelRelObsTerm(RobotInfo robot_info,
                                                        PolicyInfo policy_info,
                                                        YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = _joint_ids.size();
}

void KyonIsaacJointVelRelObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    output.resize(_joint_ids.size());
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        output(i) = inputs.v(robot_id);
    }
}

IsaacLabLastActionObsTerm::IsaacLabLastActionObsTerm(RobotInfo robot_info,
                                                     PolicyInfo policy_info,
                                                     YAML::Node config) : ObsTerm(robot_info, policy_info, config)
{
    _size = policy_info.action_size;
}

void IsaacLabLastActionObsTerm::process_impl(const Inputs& inputs, Eigen::VectorXd& output)
{
    output = inputs.last_action;
}

}
