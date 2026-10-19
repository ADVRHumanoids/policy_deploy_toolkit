#include "actuator_gains.h"

#include <iostream>

namespace {

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

void expect_error(const YAML::Node& actuators, const std::vector<std::string>& joints,
                  const std::string& message)
{
    try
    {
        XBot::policy::parse_actuator_gains(actuators, joints);
    }
    catch(const std::runtime_error& error)
    {
        require(std::string(error.what()).find(message) != std::string::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected error containing: " + message);
}

} // namespace

int main(int argc, char** argv)
{
    auto actuators = YAML::Load(R"(
legs:
  joint_names_expr: ['hip_.*', 'knee_.*']
  stiffness: 600
  damping: 10
wheels:
  joint_names_expr: ['wheel_.*']
  stiffness: 0
  damping: 5
  velocity_error_integral_gain: 6
)");
    const std::vector<std::string> joints{"wheel_left", "knee_left", "hip_right"};
    auto [stiffness, damping] = XBot::policy::parse_actuator_gains(actuators, joints);
    require(stiffness == std::vector<double>{6, 600, 600}, "Incorrect stiffness or joint order");
    require(damping == std::vector<double>{5, 10, 10}, "Incorrect damping or joint order");

    actuators["wheels"]["velocity_error_integral_gain"] = 0;
    require(XBot::policy::parse_actuator_gains(actuators, joints).first[0] == 0,
            "Zero integral gain must retain actuator stiffness");
    actuators["wheels"]["velocity_error_integral_gain"] = -6;
    require(XBot::policy::parse_actuator_gains(actuators, joints).first[0] == -6,
            "Any nonzero integral gain must replace actuator stiffness");
    actuators["wheels"]["stiffness"] = 1;
    expect_error(actuators, joints, "must have zero stiffness");
    actuators["wheels"]["stiffness"] = 0;
    actuators["wheels"]["velocity_error_integral_gain"] = YAML::Null;
    require(XBot::policy::parse_actuator_gains(actuators, joints).first[0] == 0,
            "Null optional integral gain must default to zero");

    actuators["legs"]["stiffness"] = YAML::Load("{'hip_.*': 300, 'knee_.*': 600}");
    actuators["legs"]["damping"] = YAML::Load("{'hip_.*': 8, 'knee_.*': 10}");
    auto mapped = XBot::policy::parse_actuator_gains(actuators, joints);
    require(mapped.first == std::vector<double>{0, 600, 300}, "Incorrect regex stiffness map");
    require(mapped.second == std::vector<double>{5, 10, 8}, "Incorrect regex damping map");

    const auto unactuated = XBot::policy::parse_actuator_gains(actuators, {"unmatched"});
    require(unactuated.first[0] == -1 && unactuated.second[0] == -1,
            "Unactuated joints must retain the export's -1 sentinel");
    actuators["duplicate"] = YAML::Clone(actuators["wheels"]);
    expect_error(actuators, joints, "Multiple actuators match");
    actuators.remove("duplicate");
    actuators["legs"]["stiffness"][".*"] = 1;
    expect_error(actuators, joints, "Multiple stiffness patterns match");
    actuators["legs"]["stiffness"].remove(".*");
    actuators["legs"]["stiffness"].remove("hip_.*");
    expect_error(actuators, joints, "Cannot resolve stiffness");
    expect_error(YAML::Load("[]"), joints, "Expected scene.robot.actuators");

    auto metadata = YAML::Load("scene: {}\njoint_stiffness: [1, 2, 3]\njoint_damping: [4, 5, 6]");
    const auto legacy = XBot::policy::parse_policy_gains(metadata, joints);
    require(legacy.first == std::vector<double>{1, 2, 3}, "Incorrect legacy stiffness");
    require(legacy.second == std::vector<double>{4, 5, 6}, "Incorrect legacy damping");
    metadata["scene"]["robot"]["actuators"] = YAML::Load(R"(
all:
  joint_names_expr: ['.*']
  stiffness: 0
  damping: 5
  velocity_error_integral_gain: 6
)");
    require(XBot::policy::parse_policy_gains(metadata, joints).first == std::vector<double>{6, 6, 6},
            "Articulation gains must take priority over top-level gains");

    // Optionally check a real export without depending on external files in CTest.
    if(argc > 1)
    {
        const auto metadata = YAML::LoadFile(argv[1]);
        const auto names = metadata["joint_names"].as<std::vector<std::string>>();
        const auto gains = XBot::policy::parse_policy_gains(metadata, names);
        for(std::size_t i = 0; i < names.size(); ++i)
        {
            std::cout << names[i] << ": stiffness=" << gains.first[i]
                      << " damping=" << gains.second[i] << '\n';
        }
    }
}
