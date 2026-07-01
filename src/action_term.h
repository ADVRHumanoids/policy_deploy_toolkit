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

    int size() const;

    void process(const Eigen::VectorXd& raw_action, Outputs& outputs);

    virtual void process_impl(const Eigen::VectorXd& raw_action, Outputs& outputs) = 0;

protected:

    int _size = -1;
    std::vector<int> _joint_ids;
    RobotInfo _robot_info;
    PolicyInfo _policy_info;

private:

};

class IsaacLabJointPositionActionTerm : public ActionTerm
{
public:

    IsaacLabJointPositionActionTerm(RobotInfo robot_info, 
                                    PolicyInfo policy_info,
                                    YAML::Node config);

    void process_impl(const Eigen::VectorXd& raw_action, Outputs& outputs) override;

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

    void process_impl(const Eigen::VectorXd& raw_action, Outputs& outputs) override;

private:

    double _offset;
    double _scale;
    
};

} // namespace XBot::policy

#endif // __XBOT2_DEPLOY_POLICY_ACTION_TERM_H