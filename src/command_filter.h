#pragma once

#include <Eigen/Core>
#include <cmath>
#include <numbers>
#include <optional>

namespace policy_deploy {

// Exact first-order low-pass update for a target held over dt seconds.
inline void filter_command(Eigen::VectorXd& value,
                           const Eigen::VectorXd& target,
                           std::optional<double> cutoff_hz,
                           double dt)
{
    if(!cutoff_hz)
    {
        value = target;
    }
    else if(dt > 0.0)
    {
        const double alpha = -std::expm1(-2.0 * std::numbers::pi * *cutoff_hz * dt);
        value += alpha * (target - value);
    }
}

} // namespace policy_deploy
