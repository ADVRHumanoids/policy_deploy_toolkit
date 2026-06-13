#ifndef __XBOT2_DEPLOY_POLICY_OBS_TERM_H
#define __XBOT2_DEPLOY_POLICY_OBS_TERM_H

#include "types.h"

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

class ObsTerm 
{

public:

    static std::unique_ptr<ObsTerm> create(
        std::string func,
        RobotInfo robot_info, 
        PolicyInfo policy_info,
        YAML::Node config);

    ObsTerm(RobotInfo robot_info, 
            PolicyInfo policy_info,
            YAML::Node config);

    virtual ~ObsTerm();

    int size() const;

    virtual void process_impl(const Inputs& inputs, Eigen::VectorXd& output) = 0;

    void process(const Inputs& inputs, Eigen::VectorXd& output);

protected:

    int _size = -1;
    int _n_history = 1;
    RobotInfo _robot_info; 
    PolicyInfo _policy_info;

    std::vector<int> _joint_ids;
    Eigen::VectorXd _history_buffer;

};

class IsaacLabBaseAngVelObsTerm : public ObsTerm
{

public:

    IsaacLabBaseAngVelObsTerm(RobotInfo robot_info, 
                              PolicyInfo policy_info,
                              YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;
    
};

class IsaacLabProjectedGravityObsTerm : public ObsTerm
{

public:

    IsaacLabProjectedGravityObsTerm(RobotInfo robot_info, 
                              PolicyInfo policy_info,
                              YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;

};

class IsaacLabGeneratedCommandsObsTerm : public ObsTerm
{
public:

    IsaacLabGeneratedCommandsObsTerm(RobotInfo robot_info, PolicyInfo policy_info,
                              YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;

    std::string _command_name;
    
};

class IsaacLabJointPosRelObsTerm : public ObsTerm
{
public:

    IsaacLabJointPosRelObsTerm(RobotInfo robot_info, PolicyInfo policy_info,
                               YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;

};

class KyonIsaacJointPosErrorObsTerm : public ObsTerm
{
public:

    KyonIsaacJointPosErrorObsTerm(RobotInfo robot_info, PolicyInfo policy_info,
                               YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;
};

class KyonIsaacJointVelRelObsTerm : public ObsTerm
{
public:

    KyonIsaacJointVelRelObsTerm(RobotInfo robot_info, PolicyInfo policy_info,
                               YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;
};

class IsaacLabLastActionObsTerm : public ObsTerm
{
public:

    IsaacLabLastActionObsTerm(RobotInfo robot_info, PolicyInfo policy_info,
                               YAML::Node config);

    void process_impl(const Inputs& inputs, Eigen::VectorXd& output) override;
    
};

}

#endif // __XBOT2_DEPLOY_POLICY_OBS_TERM_H
