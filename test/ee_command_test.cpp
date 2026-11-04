#include "ee_command.h"

#include <cstdio>

using namespace XBot::policy;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    if(!ok)
    {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}

int main()
{
    const Eigen::Matrix3d Ry = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d Rz = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    check(remove_yaw(Rz * Ry).isApprox(Ry, 1e-12), "remove_yaw keeps roll/pitch only");

    std::vector<Eigen::Vector3d> feet{{0.3, 0.2, -0.50}, {0.3, -0.2, -0.55}};
    check(std::abs(pelvis_height(Eigen::Matrix3d::Identity(), feet, 0.02) - 0.57) < 1e-12, "pelvis height from lowest foot");

    Eigen::Affine3d b_T_ee = Eigen::Affine3d::Identity();
    b_T_ee.translation() << 0.4, 0.2, 0.5;
    const EePose meas = to_task_frame(Eigen::Matrix3d::Identity(), 0.57, b_T_ee);
    check(meas.pos.isApprox(Eigen::Vector3d(0.4, 0.2, 1.07)), "task frame position");

    auto c = ee_command(meas, meas, meas, 0.02);
    check(c.head<6>().norm() < 1e-12, "zero twist at the target");

    EePose flipped = meas;
    flipped.quat.coeffs() *= -1.0;
    c = ee_command(flipped, meas, flipped, 0.02);
    check(c.head<6>().norm() < 1e-12, "zero twist across quaternion sign");
    check(c(9) >= 0.0, "goal quaternion w >= 0");

    EePose target = meas;
    target.pos.x() += 0.01;
    target.quat = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) * meas.quat;
    c = ee_command(meas, target, target, 0.02);
    check(c.segment<3>(0).isApprox(Eigen::Vector3d(0.5, 0.0, 0.0), 1e-9), "linear twist");
    check(c.segment<3>(3).isApprox(Eigen::Vector3d(0.0, 0.0, 5.0), 1e-9), "angular twist");

    std::printf(failures ? "ee_command_test: %d failure(s)\n" : "ee_command_test: ok\n", failures);
    return failures ? 1 : 0;
}
