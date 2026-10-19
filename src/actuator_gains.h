#pragma once

#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

// Return stiffness and damping in policy joint order, resolving actuator regexes.
inline std::pair<std::vector<double>, std::vector<double>> parse_actuator_gains(
    const YAML::Node& actuators, const std::vector<std::string>& joint_names)
{
    if(!actuators.IsMap())
    {
        throw std::runtime_error("Expected scene.robot.actuators to be a map");
    }

    // Keep the export's -1 sentinel for joints with no configured actuator.
    std::vector<double> stiffness(joint_names.size(), -1.0), damping(joint_names.size(), -1.0);
    std::vector<bool> assigned(joint_names.size(), false);

    for(const auto& entry : actuators)
    {
        const auto name = entry.first.as<std::string>();
        const auto config = entry.second;
        std::vector<std::regex> patterns;
        for(const auto& pattern : config["joint_names_expr"].as<std::vector<std::string>>())
        {
            patterns.emplace_back(pattern);
        }

        // Isaac Lab gains may be scalars or maps from joint regexes to values.
        const auto resolve_gain = [&](const char* key, const std::string& joint, bool optional = false) {
            const auto gain = config[key];
            if(optional && (!gain || gain.IsNull()))
            {
                return 0.0;
            }
            if(gain.IsScalar())
            {
                return gain.as<double>();
            }
            if(gain.IsMap())
            {
                bool matched = false;
                double value = 0.0;
                for(const auto& item : gain)
                {
                    if(std::regex_match(joint, std::regex(item.first.as<std::string>())))
                    {
                        if(matched)
                        {
                            throw std::runtime_error("Multiple " + std::string(key) +
                                " patterns match joint '" + joint + "' in actuator '" + name + "'");
                        }
                        value = item.second.as<double>();
                        matched = true;
                    }
                }
                if(matched)
                {
                    return value;
                }
            }
            throw std::runtime_error("Cannot resolve " + std::string(key) +
                " for joint '" + joint + "' in actuator '" + name + "'");
        };

        for(std::size_t i = 0; i < joint_names.size(); ++i)
        {
            bool matches = false;
            for(const auto& pattern : patterns)
            {
                matches = matches || std::regex_match(joint_names[i], pattern);
            }
            if(!matches)
            {
                continue;
            }
            if(assigned[i])
            {
                throw std::runtime_error("Multiple actuators match policy joint '" + joint_names[i] + "'");
            }

            const double act_stiffness = resolve_gain("stiffness", joint_names[i]);
            const double integral_gain = resolve_gain("velocity_error_integral_gain", joint_names[i], true);
            if(integral_gain != 0.0 && act_stiffness != 0.0)
            {
                throw std::runtime_error("Actuator '" + name + "' for joint '" + joint_names[i] +
                    "' must have zero stiffness when velocity_error_integral_gain is nonzero");
            }
            stiffness[i] = integral_gain != 0.0 ? integral_gain : act_stiffness;
            damping[i] = resolve_gain("damping", joint_names[i]);
            assigned[i] = true;
        }
    }

    return {std::move(stiffness), std::move(damping)};
}

inline std::pair<std::vector<double>, std::vector<double>> parse_policy_gains(
    const YAML::Node& metadata, const std::vector<std::string>& joint_names)
{
    if(const auto scene = metadata["scene"]; scene && scene.IsMap())
    {
        if(const auto robot = scene["robot"]; robot && robot.IsMap())
        {
            if(const auto actuators = robot["actuators"])
            {
                return parse_actuator_gains(actuators, joint_names);
            }
        }
    }

    // Older exports omit the articulation and only store resolved joint gains.
    auto stiffness = metadata["joint_stiffness"].as<std::vector<double>>();
    auto damping = metadata["joint_damping"].as<std::vector<double>>();
    if(stiffness.size() != joint_names.size() || damping.size() != joint_names.size())
    {
        throw std::runtime_error("Policy gain sizes must match joint_names");
    }
    return {std::move(stiffness), std::move(damping)};
}

} // namespace XBot::policy
