#include "action_term.h"

#include <regex>
#include <stdexcept>
#include <utility>

using namespace XBot::policy;

std::unique_ptr<ActionTerm> XBot::policy::ActionTerm::create(std::string func, 
    RobotInfo robot_info, 
    PolicyInfo policy_info, 
    YAML::Node config)
{
    if(func == "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction") 
    {
        return std::make_unique<IsaacLabJointPositionActionTerm>(robot_info, policy_info, config);
    } 
    else if(func == "isaaclab.envs.mdp.actions.joint_actions:JointVelocityAction") 
    {
        return std::make_unique<IsaacLabJointVelocityActionTerm>(robot_info, policy_info, config);
    }
    else 
    {
        throw std::runtime_error("Unsupported observation function: " + func);
    }
}

ActionTerm::ActionTerm(RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config) : _robot_info(std::move(robot_info)),
                                                                                          _policy_info(std::move(policy_info))
{
    // sanity
    if(auto n = config["clip"]; n.as<std::string>() != "null")
    {
        throw std::runtime_error("Clipping not supported for ActionTerm yet");
    }

    if(auto n = config["preserve_order"]; n.as<bool>())
    {
        throw std::runtime_error("preserve_order = true not supported for ActionTerm yet");
    }

    // resolve joint names to ids
    auto joint_names_regex = config["joint_names"].as<std::vector<std::string>>();

    std::vector<std::regex> joint_name_patterns;
    joint_name_patterns.reserve(joint_names_regex.size());
    for(const auto& joint_name_regex : joint_names_regex)
    {
        joint_name_patterns.emplace_back(joint_name_regex, std::regex::ECMAScript | std::regex::optimize);
    }

    // for(std::size_t policy_id = 0; policy_id < _policy_info.joint_names.size(); ++policy_id)
    // {
    //     const auto& joint_name = _policy_info.joint_names[policy_id];
    //     for(const auto& joint_name_pattern : joint_name_patterns)
    //     {
    //         if(std::regex_match(joint_name, joint_name_pattern))
    //         {
    //             _joint_ids.push_back(static_cast<int>(policy_id));
    //             break;
    //         }
    //     }
    // }

    for(std::size_t i = 0; i < joint_name_patterns.size(); ++i)
    {
        const auto& pattern = joint_name_patterns[i];

        int n_matching = 0;
        for(std::size_t policy_id = 0; policy_id < _policy_info.joint_names.size(); ++policy_id)
        {
            const auto& joint_name = _policy_info.joint_names[policy_id];
            if(std::regex_match(joint_name, pattern))
            {
                ++n_matching;
                _joint_ids.push_back(static_cast<int>(policy_id));
            }
        }
        
        if(n_matching == 0)
        {
            throw std::runtime_error("No joint name matches pattern: " + joint_names_regex[i]);
        }

    }
}

int ActionTerm::size() const
{
    if(_size < 0)
    {
        throw std::runtime_error("ActionTerm size not set");
    }

    return _size;
}

void ActionTerm::process(const Eigen::VectorXd &raw_action, Outputs &outputs)
{
    process_impl(raw_action, outputs);
}

IsaacLabJointPositionActionTerm::IsaacLabJointPositionActionTerm(RobotInfo robot_info, 
    PolicyInfo policy_info, YAML::Node config):
    ActionTerm(std::move(robot_info), std::move(policy_info), config)
{
    _size = _joint_ids.size();
    _offset = config["offset"].as<double>(0.0);
    _scale = config["scale"].as<double>(1.0);
}

void IsaacLabJointPositionActionTerm::process_impl(const Eigen::VectorXd &raw_action, Outputs &outputs)
{
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        outputs.q_des(robot_id) = raw_action(i) * _scale + _policy_info.joint_default_pos[policy_id] + _offset;
        outputs.k_des(robot_id) = _policy_info.stiffness[policy_id];
        outputs.d_des(robot_id) = _policy_info.damping[policy_id];
        outputs.ctrl_mode(robot_id) |= (1 + 8 + 16); // position + stiffness + damping
    }
}

IsaacLabJointVelocityActionTerm::IsaacLabJointVelocityActionTerm(RobotInfo robot_info, 
    PolicyInfo policy_info, YAML::Node config):
    ActionTerm(std::move(robot_info), std::move(policy_info), config)
{
    _size = _joint_ids.size();
    _scale = config["scale"].as<double>(1.0);
    _offset = config["offset"].as<double>(0.0);
}

void IsaacLabJointVelocityActionTerm::process_impl(const Eigen::VectorXd &raw_action, Outputs &outputs)
{
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        outputs.v_des(robot_id) = raw_action(i) * _scale + _policy_info.joint_default_vel[policy_id] + _offset;
        outputs.ctrl_mode(robot_id) |= 2; // velocity
    }
}
