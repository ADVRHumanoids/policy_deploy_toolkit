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

    void process(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs);

    virtual void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) = 0;

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

    void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) override;

private:

    double _offset;
    double _scale;
    double _alpha;
    Eigen::VectorXd _prev_raw_action;

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

// q_target = q_measured + clamp(a / sigma_a, -1, 1), q_measured read at the policy step
class KyonIsaacCurrentOffsetJointPositionActionTerm : public ActionTerm
{
public:

    KyonIsaacCurrentOffsetJointPositionActionTerm(RobotInfo robot_info,
                                                  PolicyInfo policy_info,
                                                  YAML::Node config);

    void process_impl(const Eigen::VectorXd& raw_action, const Inputs& inputs, Outputs& outputs) override;

private:

    double _sigma_a;

};

} // namespace XBot::policy

#endif // __XBOT2_DEPLOY_POLICY_ACTION_TERM_H