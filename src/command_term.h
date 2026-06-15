#ifndef __XBOT2_DEPLOY_POLICY_COMMAND_TERM_H
#define __XBOT2_DEPLOY_POLICY_COMMAND_TERM_H

#include "types.h"

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

struct CommandSpec
{
    std::string name;
    std::string class_type;
    std::vector<std::string> field_names;
    Eigen::VectorXd min;
    Eigen::VectorXd max;
    Eigen::VectorXd default_value;
};

class CommandTerm
{

public:

    static std::unique_ptr<CommandTerm> create(
        std::string name,
        std::string class_type,
        RobotInfo robot_info,
        PolicyInfo policy_info,
        YAML::Node config);

    CommandTerm(std::string name,
                std::string class_type,
                RobotInfo robot_info,
                PolicyInfo policy_info,
                YAML::Node config);

    virtual ~CommandTerm();

    int size() const;

    const CommandSpec& spec() const;

    bool sanitize(const Eigen::VectorXd& raw_command,
                  Eigen::VectorXd& sanitized_command,
                  std::string* reason = nullptr) const;

protected:

    void set_fields(std::vector<std::string> field_names,
                    Eigen::VectorXd min,
                    Eigen::VectorXd max,
                    Eigen::VectorXd default_value);

    CommandSpec _spec;
    RobotInfo _robot_info;
    PolicyInfo _policy_info;

private:

};

class KyonIsaacVelocityCommand : public CommandTerm
{

public:

    KyonIsaacVelocityCommand(RobotInfo robot_info,
                             PolicyInfo policy_info,
                             std::string name,
                             std::string class_type,
                             YAML::Node config);

private:
};

}

#endif // __XBOT2_DEPLOY_POLICY_COMMAND_TERM_H
