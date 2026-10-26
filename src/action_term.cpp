#include "action_term.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <regex>
#include <stdexcept>
#include <utility>

using namespace XBot::policy;

std::unique_ptr<ActionTerm> XBot::policy::ActionTerm::create(std::string func, 
    RobotInfo robot_info, 
    PolicyInfo policy_info, 
    YAML::Node config)
{
    if(func == "kyon_isaac.tasks.locomotion.velocity.mdp.actions:OmnisteeringAction")
    {
        return std::make_unique<OmnisteeringActionTerm>(robot_info, policy_info, config);
    }
    if(func == "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction") 
    {
        return std::make_unique<IsaacLabJointPositionActionTerm>(robot_info, policy_info, config);
    } 
    if(func == "kyon_isaac.tasks.locomotion.velocity.mdp.actions:LimitedJointPositionAction")
    {
        return std::make_unique<IsaacLabJointPositionActionTerm>(robot_info, policy_info, config);
    }
    else if(func == "isaaclab.envs.mdp.actions.joint_actions:JointVelocityAction") 
    {
        return std::make_unique<IsaacLabJointVelocityActionTerm>(robot_info, policy_info, config);
    }
    else 
    {
        throw std::runtime_error("Unsupported action function: " + func);
    }
}

ActionTerm::ActionTerm(RobotInfo robot_info, PolicyInfo policy_info) : _robot_info(std::move(robot_info)),
                                                                                          _policy_info(std::move(policy_info))
{
}

ActionTerm::ActionTerm(RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config) :
    ActionTerm(std::move(robot_info), std::move(policy_info))
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

void ActionTerm::process(const Eigen::VectorXd &raw_action, const Inputs &inputs, Outputs &outputs)
{
    process_impl(raw_action, inputs, outputs);
}

namespace {

// Match Isaac's preserve_order=True: expression order, then inventory order.
std::vector<int> resolve_ordered(const YAML::Node& node, const std::vector<std::string>& names)
{
    const auto patterns = node.as<std::vector<std::string>>();
    std::vector<int> ids;
    for(const auto& expression : patterns)
    {
        const std::regex pattern(expression);
        bool matched = false;
        for(std::size_t i = 0; i < names.size(); ++i)
        {
            if(!std::regex_match(names[i], pattern)) continue;
            if(std::find(ids.begin(), ids.end(), i) != ids.end())
                throw std::runtime_error("Omnisteering patterns match the same entry twice: " + names[i]);
            ids.push_back(static_cast<int>(i));
            matched = true;
        }
        if(!matched) throw std::runtime_error("Omnisteering pattern has no matches: " + expression);
    }
    return ids;
}

double feasible_reference(double position, double delta, double low, double high)
{
    const double direct = position + delta;
    double selected = std::clamp(direct, low, high);
    double best_travel = std::numeric_limits<double>::infinity();
    // Same candidates, feasibility test, tie order and fallback as Isaac.
    for(int turn = -2; turn <= 2; ++turn)
    {
        const double candidate = direct + turn * std::numbers::pi;
        const double travel = std::abs(candidate - position);
        if(candidate >= low && candidate <= high && travel < best_travel)
        {
            selected = candidate;
            best_travel = travel;
        }
    }
    return selected;
}

} // namespace

OmnisteeringActionTerm::OmnisteeringActionTerm(RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config) :
    ActionTerm(std::move(robot_info), std::move(policy_info)),
    _command_name(config["cmd_name"].as<std::string>("base_velocity")),
    _vmax(config["vmax"].as<double>(2.0)),
    _wheel_radius(config["wheel_radius"].as<double>(0.107)),
    _min_wheel_speed(config["min_wheel_speed"].as<double>(0.1))
{
    _size = 0;
    if(!std::isfinite(_vmax) || _vmax <= 0 || !std::isfinite(_wheel_radius) || _wheel_radius <= 0 ||
       !std::isfinite(_min_wheel_speed) || _min_wheel_speed < 0 ||
       !std::isfinite(_policy_info.control_dt) || _policy_info.control_dt <= 0)
        throw std::runtime_error("Omnisteering requires positive finite vmax, wheel_radius and control_dt, and nonnegative min_wheel_speed");
    if(!_robot_info.body_pose_in_base)
        throw std::runtime_error("Omnisteering requires live body_pose_in_base kinematics");
    const auto command = _policy_info.command_size.find(_command_name);
    if(command == _policy_info.command_size.end() || command->second < 3)
        throw std::runtime_error("Omnisteering requires a planar velocity command: " + _command_name);
    _steering_ids = resolve_ordered(config["steering_joint_names"], _policy_info.joint_names);
    _drive_ids = resolve_ordered(config["drive_joint_names"], _policy_info.joint_names);
    const auto bodies = resolve_ordered(config["wheel_names"], _robot_info.body_names);
    if(bodies.empty() || _steering_ids.size() != bodies.size() || _drive_ids.size() != bodies.size())
        throw std::runtime_error("Omnisteering requires one steering joint and one drive joint per wheel body");
    for(int id : _steering_ids)
        if(std::find(_drive_ids.begin(), _drive_ids.end(), id) != _drive_ids.end())
            throw std::runtime_error("Omnisteering steering and drive joints must be distinct");
    for(const auto& ids : {_steering_ids, _drive_ids})
    {
        for(int id : ids)
        {
            const int rid = _policy_info.joint_id_policy_to_robot.at(id);
            if(rid < 0 || rid >= _robot_info.joint_vel_max.size())
                throw std::runtime_error("Omnisteering joint has no robot mapping: " + _policy_info.joint_names[id]);
            const double limit = _robot_info.joint_vel_max[rid];
            if(!std::isfinite(limit) || limit < 0)
                throw std::runtime_error("Invalid omnisteering joint velocity limit");
        }
    }
    for(int id : _steering_ids)
    {
        const int rid = _policy_info.joint_id_policy_to_robot[id];
        const double low = _robot_info.joint_pos_min[rid], high = _robot_info.joint_pos_max[rid];
        if(!std::isfinite(low) || !std::isfinite(high) || low > high)
            throw std::runtime_error("Invalid omnisteering joint position limits");
    }
    for(int id : bodies) _wheel_names.push_back(_robot_info.body_names[id]);
    _steering_target.resize(bodies.size());
}

void OmnisteeringActionTerm::process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs)
{
    const auto& command = inputs.command.at(_command_name);
    if(raw_action.size() != 0 || command.size() < 3 || !command.head<3>().allFinite())
        throw std::runtime_error("Omnisteering requires no policy actions and a finite planar velocity command");
    // Stage all results before modifying outputs, including on kinematics failure.
    Eigen::VectorXd steering(_wheel_names.size()), drive(_wheel_names.size());
    for(std::size_t i = 0; i < _wheel_names.size(); ++i)
    {
        const int sid = _policy_info.joint_id_policy_to_robot[_steering_ids[i]];
        const auto pose = _robot_info.body_pose_in_base(_wheel_names[i]);
        if(!pose.matrix().allFinite() || !std::isfinite(inputs.q[sid]))
            throw std::runtime_error("Nonfinite omnisteering wheel pose or steering position");
        const Eigen::Vector2d velocity(command[0] - command[2] * pose.translation().y(),
                                       command[1] + command[2] * pose.translation().x());
        // Kyon's USD and deployment URDF both use wheel-local +Z as spin axis.
        Eigen::Vector2d axis = pose.linear().col(2).head<2>();
        const double norm = axis.norm();
        axis /= std::max(norm, 1.e-6);
        const Eigen::Vector2d roll(axis.y(), -axis.x());
        const bool moving = velocity.norm() > _min_wheel_speed && norm > 0.1;
        const double low = _robot_info.joint_pos_min[sid], high = _robot_info.joint_pos_max[sid];
        const double position = inputs.q[sid];
        const double previous = (!_initialized || inputs.step == 0) ? position : _steering_target[i];
        const double delta = std::atan2(roll.x() * velocity.y() - roll.y() * velocity.x(), roll.dot(velocity));
        const double reference = moving ? feasible_reference(position, delta, low, high) : previous;
        const double max_step = std::min(_vmax, double(_robot_info.joint_vel_max[sid])) * _policy_info.control_dt;
        steering[i] = std::clamp(previous + std::clamp(reference - previous, -max_step, max_step), low, high);
        const int did = _policy_info.joint_id_policy_to_robot[_drive_ids[i]];
        const double speed_limit = _robot_info.joint_vel_max[did];
        drive[i] = std::clamp(moving ? velocity.dot(roll) / _wheel_radius : 0., -speed_limit, speed_limit);
    }
    _steering_target = steering;
    _initialized = true;
    for(std::size_t i = 0; i < _wheel_names.size(); ++i)
    {
        const int spid = _steering_ids[i], dpid = _drive_ids[i];
        const int sid = _policy_info.joint_id_policy_to_robot[spid], did = _policy_info.joint_id_policy_to_robot[dpid];
        outputs.q_des[sid] = steering[i];
        outputs.k_des[sid] = _policy_info.stiffness[spid];
        outputs.d_des[sid] = _policy_info.damping[spid];
        outputs.ctrl_mode[sid] = 1 + 8 + 16;
        outputs.v_des[did] = drive[i];
        outputs.k_des[did] = _policy_info.stiffness[dpid];
        outputs.d_des[did] = _policy_info.damping[dpid];
        outputs.ctrl_mode[did] = 2 + 8 + 16;
    }
}

IsaacLabJointPositionActionTerm::IsaacLabJointPositionActionTerm(RobotInfo robot_info, 
    PolicyInfo policy_info, YAML::Node config):
    ActionTerm(std::move(robot_info), std::move(policy_info), config)
{
    _size = _joint_ids.size();
    _offset = config["offset"].as<double>(0.0);
    _scale = config["scale"].as<double>(1.0);
}

void IsaacLabJointPositionActionTerm::process_impl(const Eigen::VectorXd &raw_action, const Inputs &inputs, Outputs &outputs)
{
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        if(robot_id < 0)
        {
            throw std::runtime_error(
                "JointPositionAction references policy joint '" + _policy_info.joint_names[policy_id] + "' with no robot mapping");
        }

        // clamp to joint position limits
        float q_des = raw_action(i) * _scale + _policy_info.joint_default_pos[policy_id] + _offset;
        float q_des_clamped = std::clamp(q_des, 
            _robot_info.joint_pos_min(robot_id), 
            _robot_info.joint_pos_max(robot_id)
        );

        // clamp to joint velocity limits
        float v_des = (q_des_clamped - outputs.q_des(robot_id)) / _policy_info.control_dt;
        float v_des_clamped = std::clamp(v_des,
            -_robot_info.joint_vel_max(robot_id),
            _robot_info.joint_vel_max(robot_id) 
        );
        q_des_clamped = outputs.q_des(robot_id) + v_des_clamped * _policy_info.control_dt;

        outputs.q_des(robot_id) = q_des_clamped;
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

void IsaacLabJointVelocityActionTerm::process_impl(const Eigen::VectorXd &raw_action, const Inputs &inputs, Outputs &outputs)
{
    for(std::size_t i = 0; i < _joint_ids.size(); ++i)
    {
        int policy_id = _joint_ids[i];
        int robot_id = _policy_info.joint_id_policy_to_robot[policy_id];
        if(robot_id < 0)
        {
            throw std::runtime_error(
                "JointVelocityAction references policy joint '" + _policy_info.joint_names[policy_id] + "' with no robot mapping");
        }

        float v_des = raw_action(i) * _scale + _policy_info.joint_default_vel[policy_id] + _offset;
        float v_des_clamped = std::clamp(v_des,
            -_robot_info.joint_vel_max(robot_id),
            _robot_info.joint_vel_max(robot_id) 
        );
        outputs.v_des(robot_id) = v_des_clamped;
        outputs.k_des(robot_id) = _policy_info.stiffness[policy_id];
        outputs.d_des(robot_id) = _policy_info.damping[policy_id];
        outputs.ctrl_mode(robot_id) |= (2 + 8 + 16); // velocity + stiffness + damping
    }
}
