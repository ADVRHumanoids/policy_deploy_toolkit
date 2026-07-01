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

    bool sanitize(const Inputs& inputs, 
                  const Eigen::VectorXd& raw_command,
                  Eigen::VectorXd& sanitized_command,
                  std::string* reason = nullptr) const;

protected:

    void set_fields(std::vector<std::string> field_names,
                    Eigen::VectorXd min,
                    Eigen::VectorXd max,
                    Eigen::VectorXd default_value);

    virtual bool post_process(const Inputs& inputs, 
                             const Eigen::VectorXd& command,
                             Eigen::VectorXd& post_processed_command,
                             std::string* reason = nullptr) const
    {
        // Default implementation: no additional checks
        return true;
    }

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

class KyonIsaacTerrainBasedVelocityCommand : public KyonIsaacVelocityCommand
{
public:

    KyonIsaacTerrainBasedVelocityCommand(RobotInfo robot_info,
                                         PolicyInfo policy_info,
                                         std::string name,
                                         std::string class_type,
                                         YAML::Node config);

    bool post_process(const Inputs& inputs, 
                      const Eigen::VectorXd& command,
                      Eigen::VectorXd& post_processed_command,
                      std::string* reason = nullptr) const override;

private:
    double compute_terrain_difficulty(const Eigen::VectorXd height_scan) const;

    YAML::Node _config;
};

}

#endif // __XBOT2_DEPLOY_POLICY_COMMAND_TERM_H
