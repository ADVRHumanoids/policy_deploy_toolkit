#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

inline std::vector<std::string> actor_observation_groups(
    const YAML::Node& metadata, const std::string& override)
{
    if(!override.empty())
    {
        return {override};
    }

    const auto agent = metadata["agent"];
    const auto groups = agent && agent.IsMap() ? agent["obs_groups"] : YAML::Node{};
    const auto policy = groups && groups.IsMap() ? groups["policy"] : YAML::Node{};
    if(policy.IsDefined() && !policy.IsNull())
    {
        if(!policy.IsSequence() || policy.size() == 0)
        {
            throw std::runtime_error("Expected agent.obs_groups.policy to be a non-empty list");
        }
        return policy.as<std::vector<std::string>>();
    }

    // Compatibility with exports predating agent.obs_groups.
    if(const auto legacy = metadata["default_obs_group"])
    {
        return {legacy.as<std::string>()};
    }
    return {"policy"};
}

} // namespace XBot::policy
