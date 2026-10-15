#ifndef __DEPLOY_POLICY_TOOLKIT_UTILS_H
#define __DEPLOY_POLICY_TOOLKIT_UTILS_H

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <format>

namespace XBot::policy {

template<typename T>
T yaml_get_or_throw(YAML::Node node, std::vector<std::string> keys, const std::string& id)
{
    // full path for error message
    std::string full_path = keys[0];
    for(std::size_t i = 1; i < keys.size(); ++i)
    {
        full_path += "." + keys[i];
    }

    auto do_throw = [&]() {
        throw std::runtime_error(
            std::format("Failed to parse '{}' with type '{}' in policy description for '{}'", 
                full_path, typeid(T).name(), id)
        );
    };

    for(auto key : keys)
    {
        if(!node.IsMap())
        {
            do_throw();
        }

        node = node[key];

        if(!node)
        {
            do_throw();
        }
    }

    try
    {
        return node.as<T>();
    }
    catch(const YAML::BadConversion&)
    {
        do_throw();
    }
}

} // namespace XBot::policy


#endif // __DEPLOY_POLICY_TOOLKIT_UTILS_H