#ifndef __XBOT2_DEPLOY_POLICY_EE_COMMAND_H
#define __XBOT2_DEPLOY_POLICY_EE_COMMAND_H

#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <vector>

namespace XBot::policy {

// EE pose in the task frame: origin on the ground under the pelvis, z up, x along the base heading.
struct EePose {
    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()};
};

// Base orientation with its yaw removed (R = Rz Ry Rx): Rz(-yaw) * w_R_b.
inline Eigen::Matrix3d remove_yaw(const Eigen::Matrix3d& w_R_b)
{
    const double yaw = std::atan2(w_R_b(1, 0), w_R_b(0, 0));
    return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * w_R_b;
}

// Pelvis height above flat ground from the lowest foot frame, which stands foot_contact_height above the ground.
inline double pelvis_height(const Eigen::Matrix3d& t_R_b,
                            const std::vector<Eigen::Vector3d>& b_p_feet,
                            double foot_contact_height)
{
    double h = -std::numeric_limits<double>::infinity();
    for(const auto& p : b_p_feet)
    {
        h = std::max(h, -(t_R_b * p).z());
    }
    return h + foot_contact_height;
}

inline EePose to_task_frame(const Eigen::Matrix3d& t_R_b, double h, const Eigen::Affine3d& b_T_ee)
{
    EePose out;
    out.pos = t_R_b * b_T_ee.translation() + Eigen::Vector3d(0.0, 0.0, h);
    out.quat = Eigen::Quaterniond(t_R_b * b_T_ee.linear()).normalized();
    return out;
}

inline Eigen::Quaterniond quat_unique(Eigen::Quaterniond q)
{
    if(q.w() < 0.0)
    {
        q.coeffs() *= -1.0;
    }
    return q;
}

// Per arm: twist towards the target (v, w), goal position, goal quaternion wxyz with w >= 0.
inline Eigen::Matrix<double, 13, 1> ee_command(const EePose& meas, const EePose& target, const EePose& goal, double dt)
{
    const Eigen::Quaterniond q_tgt = quat_unique(target.quat.normalized());
    const Eigen::Quaterniond q_goal = quat_unique(goal.quat.normalized());
    const Eigen::AngleAxisd err(quat_unique(q_tgt * meas.quat.conjugate()));

    Eigen::Matrix<double, 13, 1> c;
    c.segment<3>(0) = (target.pos - meas.pos) / dt;
    c.segment<3>(3) = err.angle() * err.axis() / dt;
    c.segment<3>(6) = goal.pos;
    c.segment<4>(9) << q_goal.w(), q_goal.x(), q_goal.y(), q_goal.z();
    return c;
}

}

#endif // __XBOT2_DEPLOY_POLICY_EE_COMMAND_H
