#include "command_term.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace {

constexpr auto kVelocityCommandClass =
    "kyon_isaac.tasks.locomotion.velocity.mdp.commands:VelocityCommand";

std::unique_ptr<XBot::policy::CommandTerm> make_velocity_command()
{
    auto config = YAML::Load(R"(
ranges:
  lin_vel_x: [-2.0, 2.0]
  lin_vel_y: [-1.0, 1.0]
  ang_vel_z: [-1.0, 1.0]
)");

    return XBot::policy::CommandTerm::create(
        "base_velocity",
        kVelocityCommandClass,
        XBot::policy::RobotInfo{},
        XBot::policy::PolicyInfo{},
        config);
}

bool approx(double lhs, double rhs)
{
    return std::abs(lhs - rhs) < 1e-12;
}

} // namespace

int main()
{
    auto command = make_velocity_command();
    const auto& spec = command->spec();

    assert(command->size() == 3);
    assert(spec.name == "base_velocity");
    assert(spec.class_type == kVelocityCommandClass);
    assert((spec.field_names == std::vector<std::string>{"lin_vel_x", "lin_vel_y", "ang_vel_z"}));
    assert(spec.default_value.isZero());

    Eigen::VectorXd raw(3);
    Eigen::VectorXd sanitized;
    std::string reason;

    raw << 0.5, -0.25, 0.75;
    assert(command->sanitize(raw, sanitized, &reason));
    assert(reason.empty());
    assert(approx(sanitized(0), 0.5));
    assert(approx(sanitized(1), -0.25));
    assert(approx(sanitized(2), 0.75));

    raw << 10.0, -10.0, 5.0;
    assert(command->sanitize(raw, sanitized, &reason));
    assert(approx(sanitized(0), 2.0));
    assert(approx(sanitized(1), -1.0));
    assert(approx(sanitized(2), 1.0));

    Eigen::VectorXd wrong_size(2);
    wrong_size << 0.0, 0.0;
    assert(!command->sanitize(wrong_size, sanitized, &reason));
    assert(!reason.empty());

    raw << 0.0, std::nan(""), 0.0;
    assert(!command->sanitize(raw, sanitized, &reason));
    assert(!reason.empty());

    return 0;
}
