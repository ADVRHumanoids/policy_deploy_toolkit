#ifndef __XBOT2_DEPLOY_POLICY_ACTION_TERM_H
#define __XBOT2_DEPLOY_POLICY_ACTION_TERM_H

#include "types.h"

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

class ActionTerm
{

public:

    static std::unique_ptr<ActionTerm> create(
        std::string func,
        RobotInfo robot_info, 
        PolicyInfo policy_info,
        YAML::Node config);

    ActionTerm(RobotInfo robot_info, 
               PolicyInfo policy_info,
               YAML::Node config);

    virtual ~ActionTerm() = default;

    int size() const;

    void process(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs);

    virtual void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) = 0;

protected:

    ActionTerm(RobotInfo robot_info, PolicyInfo policy_info);

    int _size = -1;
    std::vector<int> _joint_ids;
    RobotInfo _robot_info;
    PolicyInfo _policy_info;

private:

};

class OmnisteeringActionTerm : public ActionTerm
{
public:
    OmnisteeringActionTerm(RobotInfo robot_info, PolicyInfo policy_info, YAML::Node config);
    void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) override;

private:
    std::vector<int> _steering_ids, _drive_ids;
    std::vector<std::string> _wheel_names;
    std::string _command_name;
    double _vmax, _wheel_radius, _min_wheel_speed;
    Eigen::VectorXd _steering_target;
    bool _initialized = false;
};

class IsaacLabJointPositionActionTerm : public ActionTerm
{
public:

    IsaacLabJointPositionActionTerm(RobotInfo robot_info, 
                                    PolicyInfo policy_info,
                                    YAML::Node config);

    void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) override;

private:

    double _offset;
    double _scale;

};

class IsaacLabJointVelocityActionTerm : public ActionTerm
{
public:

    IsaacLabJointVelocityActionTerm(RobotInfo robot_info, 
                                    PolicyInfo policy_info,
                                    YAML::Node config);

    void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) override;

private:

    double _offset;
    double _scale;
    
};

} // namespace XBot::policy

#endif // __XBOT2_DEPLOY_POLICY_ACTION_TERM_H
