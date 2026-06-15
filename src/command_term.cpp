#include "command_term.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <set>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace {

std::pair<double, double> parse_range(const YAML::Node& ranges,
                                      const std::string& command_name,
                                      const std::string& field_name)
{
    const auto range = ranges[field_name];
    if(!range)
    {
        throw std::runtime_error(
            std::format("Command '{}' is missing range '{}'", command_name, field_name));
    }

    auto values = range.as<std::vector<double>>();
    if(values.size() != 2)
    {
        throw std::runtime_error(
            std::format("Command '{}' range '{}' must contain exactly two values",
                        command_name,
                        field_name));
    }

    if(!std::isfinite(values[0]) || !std::isfinite(values[1]))
    {
        throw std::runtime_error(
            std::format("Command '{}' range '{}' contains a non-finite value",
                        command_name,
                        field_name));
    }

    if(values[0] > values[1])
    {
        throw std::runtime_error(
            std::format("Command '{}' range '{}' has min greater than max",
                        command_name,
                        field_name));
    }

    return {values[0], values[1]};
}

void reject_unknown_ranges(const YAML::Node& ranges,
                           const std::string& command_name,
                           const std::vector<std::string>& expected_fields)
{
    std::set<std::string> expected(expected_fields.begin(), expected_fields.end());
    for(const auto& pair : ranges)
    {
        auto field_name = pair.first.as<std::string>();
        if(!expected.contains(field_name))
        {
            throw std::runtime_error(
                std::format("Command '{}' has unsupported range '{}'", command_name, field_name));
        }
    }
}

} // namespace

namespace XBot::policy {

std::unique_ptr<CommandTerm> CommandTerm::create(std::string name,
                                                 std::string class_type,
                                                 RobotInfo robot_info,
                                                 PolicyInfo policy_info,
                                                 YAML::Node config)
{
    if(class_type == "kyon_isaac.tasks.locomotion.velocity.mdp.commands:VelocityCommand")
    {
        return std::make_unique<KyonIsaacVelocityCommand>(
            std::move(robot_info),
            std::move(policy_info),
            std::move(name),
            std::move(class_type),
            config);
    }

    throw std::runtime_error("Unsupported command class: " + class_type);
}

CommandTerm::CommandTerm(std::string name,
                         std::string class_type,
                         RobotInfo robot_info,
                         PolicyInfo policy_info,
                         YAML::Node /* config */):
    _robot_info(std::move(robot_info)),
    _policy_info(std::move(policy_info))
{
    _spec.name = std::move(name);
    _spec.class_type = std::move(class_type);
}

CommandTerm::~CommandTerm() = default;

int CommandTerm::size() const
{
    return static_cast<int>(_spec.field_names.size());
}

const CommandSpec& CommandTerm::spec() const
{
    return _spec;
}

bool CommandTerm::sanitize(const Eigen::VectorXd& raw_command,
                           Eigen::VectorXd& sanitized_command,
                           std::string* reason) const
{
    if(raw_command.size() != size())
    {
        if(reason)
        {
            *reason = std::format("Command '{}' expected size {}, got {}",
                                  _spec.name,
                                  size(),
                                  raw_command.size());
        }
        return false;
    }

    sanitized_command.resize(size());
    for(int i = 0; i < raw_command.size(); ++i)
    {
        const auto value = raw_command(i);
        if(!std::isfinite(value))
        {
            if(reason)
            {
                *reason = std::format("Command '{}' field '{}' is not finite",
                                      _spec.name,
                                      _spec.field_names.at(i));
            }
            return false;
        }

        sanitized_command(i) = std::clamp(value, _spec.min(i), _spec.max(i));
    }

    if(reason)
    {
        reason->clear();
    }

    return true;
}

void CommandTerm::set_fields(std::vector<std::string> field_names,
                             Eigen::VectorXd min,
                             Eigen::VectorXd max,
                             Eigen::VectorXd default_value)
{
    const auto size = static_cast<int>(field_names.size());
    if(min.size() != size || max.size() != size || default_value.size() != size)
    {
        throw std::runtime_error(
            std::format("Command '{}' has inconsistent field vector sizes", _spec.name));
    }

    _spec.field_names = std::move(field_names);
    _spec.min = std::move(min);
    _spec.max = std::move(max);
    _spec.default_value = std::move(default_value);
}

KyonIsaacVelocityCommand::KyonIsaacVelocityCommand(RobotInfo robot_info,
                                                   PolicyInfo policy_info,
                                                   std::string name,
                                                   std::string class_type,
                                                   YAML::Node config):
    CommandTerm(std::move(name), std::move(class_type), std::move(robot_info), std::move(policy_info), config)
{
    auto ranges = config["ranges"];
    if(!ranges || !ranges.IsMap())
    {
        throw std::runtime_error(
            std::format("Command '{}' must define a ranges map", _spec.name));
    }
    
    std::vector<std::string> fields;
    for(auto field : ranges)
    {
        fields.push_back(field.first.as<std::string>());
    }

    reject_unknown_ranges(ranges, _spec.name, fields);

    Eigen::VectorXd min(fields.size());
    Eigen::VectorXd max(fields.size());
    for(std::size_t i = 0; i < fields.size(); ++i)
    {
        const auto [range_min, range_max] = parse_range(ranges, _spec.name, fields[i]);
        min(static_cast<int>(i)) = range_min;
        max(static_cast<int>(i)) = range_max;

        // print the parsed range for debugging
        std::cout << std::format("Command '{}', field '{}': min={}, max={}\n", 
            _spec.name, fields[i], range_min, range_max);
    }

    set_fields(fields, std::move(min), std::move(max), Eigen::VectorXd::Zero(fields.size()));
}

} // namespace XBot::policy
