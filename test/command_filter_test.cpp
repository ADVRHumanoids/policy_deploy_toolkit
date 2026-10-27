#include "command_filter.h"

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {

void check(bool condition, const char* message)
{
    if(!condition)
        throw std::runtime_error(message);
}

} // namespace

int main()
{
    const Eigen::VectorXd target = (Eigen::Vector3d() << 1., -2., 0.5).finished();
    Eigen::VectorXd value = Eigen::Vector3d::Zero();
    policy_deploy::filter_command(value, target, std::nullopt, 0.0);
    check(value == target, "Absent cutoff must pass commands through immediately");

    value.setZero();
    const double cutoff = 2.0;
    const double tau = 1.0 / (2.0 * std::numbers::pi * cutoff);
    policy_deploy::filter_command(value, target, cutoff, tau);
    check(value.isApprox((1.0 - std::exp(-1.0)) * target), "Incorrect step response");

    Eigen::VectorXd split = Eigen::Vector3d::Zero();
    for(const double fraction : {0.1, 0.3, 0.6})
        policy_deploy::filter_command(split, target, cutoff, fraction * tau);
    check(value.isApprox(split), "Response must account for variable loop intervals");

    policy_deploy::filter_command(split, target, cutoff, 0.0);
    check(value.isApprox(split), "Zero elapsed time must hold the filtered command");
    policy_deploy::filter_command(split, target, cutoff, -1.0);
    check(value.isApprox(split), "Negative elapsed time must not integrate backwards");

    for(int i = 0; i < 100; ++i)
        policy_deploy::filter_command(value, target, cutoff, tau);
    check(value.isApprox(target), "Filter must converge to the target");

    // A timeout resets the receiver's filter state to its default (zero).
    value.setZero();
    policy_deploy::filter_command(value, target, cutoff, tau);
    check(value.isApprox((1.0 - std::exp(-1.0)) * target), "Restart must ramp from defaults");
    std::cout << "Command filter checks passed\n";
}
