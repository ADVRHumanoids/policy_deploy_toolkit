#include "action_term.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace XBot::policy;
constexpr double pi = std::numbers::pi;

void check(bool value, const char* message)
{
    if(!value) throw std::runtime_error(message);
}

void near(double value, double expected)
{
    if(std::abs(value - expected) > 2.e-6)
        throw std::runtime_error("Expected " + std::to_string(expected) + ", got " + std::to_string(value));
}

template<class F> void rejects(F&& function)
{
    try { function(); }
    catch(const std::exception&) { return; }
    throw std::runtime_error("Expected invalid configuration/input to be rejected");
}

struct Fixture
{
    RobotInfo robot;
    PolicyInfo policy{};
    Inputs inputs;
    Outputs outputs{0, 8};
    Eigen::Vector4d offsets = Eigen::Vector4d::Zero();
    bool vertical = false;
    YAML::Node config = YAML::Load(R"(
steering_joint_names: [steer_1, steer_2, steer_3, steer_4]
drive_joint_names: [drive_1, drive_2, drive_3, drive_4]
wheel_names: [wheel_1, wheel_2, wheel_3, wheel_4]
cmd_name: base_velocity
wheel_radius: 0.1
vmax: 2.0
min_wheel_speed: 0.1
)");

    Fixture()
    {
        policy.control_dt = .01;
        policy.command_size["base_velocity"] = 3;
        policy.stiffness.assign(8, 100.);
        policy.damping.assign(8, 5.);
        policy.joint_default_pos.assign(8, 0.);
        policy.joint_default_vel.assign(8, 0.);
        for(int i = 1; i <= 4; ++i) policy.joint_names.push_back("steer_" + std::to_string(i));
        for(int i = 1; i <= 4; ++i) policy.joint_names.push_back("drive_" + std::to_string(i));
        robot.joint_names.assign(policy.joint_names.rbegin(), policy.joint_names.rend());
        for(int i = 0; i < 8; ++i) policy.joint_id_policy_to_robot.push_back(7 - i);
        robot.joint_pos_min = Eigen::VectorXf::Constant(8, -2.);
        robot.joint_pos_max = Eigen::VectorXf::Constant(8, 2.);
        robot.joint_vel_max = Eigen::VectorXf::Constant(8, 100.);
        inputs.q = Eigen::VectorXd::Zero(8);
        inputs.command["base_velocity"] = Eigen::Vector3d(1., 0., 0.);
        // Inventory order is deliberately different from configured expression order.
        robot.body_names = {"wheel_4", "wheel_2", "wheel_1", "wheel_3"};
        robot.body_pose_in_base = [this](const std::string& name) -> Eigen::Affine3d {
            const int i = name.back() - '1';
            const double angle = inputs.q[sid(i)] + offsets[i];
            Eigen::Affine3d pose = Eigen::Affine3d::Identity();
            pose.translation() << (i < 2 ? .4 : -.4), (i % 2 == 0 ? .3 : -.3), 0.;
            if(!vertical)
                pose.linear() = (Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()) *
                                 Eigen::AngleAxisd(-pi / 2, Eigen::Vector3d::UnitX())).toRotationMatrix();
            return pose;
        };
    }

    int sid(int i) const { return 7 - i; }
    int did(int i) const { return 3 - i; }
    auto make()
    {
        return ActionTerm::create("kyon_isaac.tasks.locomotion.velocity.mdp.actions:OmnisteeringAction", robot, policy, config);
    }
    void step(ActionTerm& term)
    {
        term.process(Eigen::VectorXd(0), inputs, outputs);
        ++inputs.step;
    }
};

int main()
{
    // Forward/reverse, mirrored axle signs, mapping and controller modes/gains.
    {
        Fixture f;
        f.offsets << 0., pi, 0., pi;
        auto term = f.make();
        check(term->size() == 0, "Omnisteering must consume zero policy actions");
        f.step(*term);
        for(int i = 0; i < 4; ++i)
        {
            near(f.outputs.q_des[f.sid(i)], 0.);
            near(f.outputs.v_des[f.did(i)], i % 2 ? -10. : 10.);
            check(f.outputs.ctrl_mode[f.sid(i)] == 25, "Incorrect steering control mode");
            check(f.outputs.ctrl_mode[f.did(i)] == 26, "Incorrect drive control mode");
            near(f.outputs.k_des[f.sid(i)], 100.);
            near(f.outputs.d_des[f.did(i)], 5.);
        }
        f.inputs.command["base_velocity"] *= -1;
        f.step(*term);
        near(f.outputs.v_des[f.did(0)], -10.);
    }
    // Rate limiting uses the previous commanded target even if measured joints lag.
    {
        Fixture f;
        f.inputs.command["base_velocity"] = Eigen::Vector3d(std::cos(1.4), std::sin(1.4), 0.);
        auto term = f.make();
        for(int i = 1; i <= 5; ++i)
        {
            f.step(*term);
            near(f.outputs.q_des[f.sid(0)], .02 * i);
            near(f.outputs.v_des[f.did(0)], 10 * std::cos(1.4));
        }
        f.inputs.command["base_velocity"].setZero();
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], .1);
        near(f.outputs.v_des[f.did(0)], 0.);
        // A restarted policy initializes targets from measurement, as in Isaac reset.
        f.inputs.q[f.sid(0)] = -.7;
        f.inputs.step = 0;
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], -.7);
    }
    // Nearest feasible candidate, including pi-flip when the direct target is beyond a limit.
    for(double high : {2., 1.85})
    {
        Fixture f;
        f.robot.joint_pos_max.setConstant(high);
        f.inputs.q.setConstant(1.8);
        f.inputs.command["base_velocity"] = Eigen::Vector3d(std::cos(1.9), std::sin(1.9), 0.);
        auto term = f.make();
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], high == 2. ? 1.82 : 1.78);
        for(int n = 0; n < 200; ++n)
        {
            for(int i = 0; i < 4; ++i) f.inputs.q[f.sid(i)] = f.outputs.q_des[f.sid(i)];
            f.step(*term);
        }
        near(f.outputs.q_des[f.sid(0)], high == 2. ? 1.9 : 1.9 - pi);
        near(f.outputs.v_des[f.did(0)], high == 2. ? 10. : -10.);
    }
    // Narrow limits with no exact heading, pure yaw and speed saturation.
    {
        Fixture f;
        f.robot.joint_pos_min.setConstant(-.2);
        f.robot.joint_pos_max.setConstant(.2);
        f.inputs.command["base_velocity"] = Eigen::Vector3d(0., 1., 0.);
        auto term = f.make();
        for(int i = 0; i < 30; ++i) f.step(*term);
        near(f.outputs.q_des[f.sid(0)], .2);
    }
    {
        Fixture f;
        f.robot.joint_vel_max.setConstant(2.5);
        f.inputs.command["base_velocity"] = Eigen::Vector3d(0., 0., 1.);
        auto term = f.make();
        f.step(*term);
        for(int i = 0; i < 4; ++i) near(f.outputs.v_des[f.did(i)], i % 2 ? 2.5 : -2.5);
        f.vertical = true;
        const double previous = f.outputs.q_des[f.sid(0)];
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], previous);
        check(f.outputs.v_des.isZero(), "Invalid axle must disable drive");
    }
    // Invalid metadata and runtime command/pose must fail before updating outputs.
    {
        Fixture f;
        f.robot.joint_vel_max[f.sid(0)] = .5;
        f.inputs.command["base_velocity"] = Eigen::Vector3d(std::cos(1.4), std::sin(1.4), 0.);
        auto term = f.make();
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], .005);
        f.inputs.command["base_velocity"] = Eigen::Vector3d(.05, 0., 0.);
        f.step(*term);
        near(f.outputs.q_des[f.sid(0)], .005);
        check(f.outputs.v_des.isZero(), "Below-threshold command must stop rolling");
    }
    {
        Fixture f;
        f.config["vmax"] = 0.;
        rejects([&] { f.make(); });
        f.config["vmax"] = 2.;
        f.config["wheel_radius"] = -1.;
        rejects([&] { f.make(); });
        f.config["wheel_radius"] = .1;
        f.config["steering_joint_names"] = std::vector<std::string>{"steer_1", "steer_1"};
        rejects([&] { f.make(); });
    }
    {
        Fixture f;
        f.policy.joint_id_policy_to_robot[0] = -1;
        rejects([&] { f.make(); });
    }
    {
        Fixture f;
        f.robot.body_pose_in_base = {};
        rejects([&] { f.make(); });
    }
    {
        Fixture f;
        f.config["wheel_names"] = std::vector<std::string>{"missing_wheel"};
        rejects([&] { f.make(); });
    }
    {
        Fixture f;
        auto term = f.make();
        f.inputs.command["base_velocity"][0] = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { f.step(*term); });
        check(f.outputs.q_des.isZero() && f.outputs.v_des.isZero(), "Invalid input modified outputs");
    }
    // Existing learned position/velocity terms still use their joint-action metadata.
    for(bool position : {true, false})
    {
        Fixture f;
        auto config = YAML::Load("clip: null\npreserve_order: false\njoint_names: [steer_1]\nscale: 0.5");
        const auto type = position ? "JointPositionAction" : "JointVelocityAction";
        auto term = ActionTerm::create(std::string("isaaclab.envs.mdp.actions.joint_actions:") + type,
                                       f.robot, f.policy, config);
        check(term->size() == 1, "Legacy action dimension changed");
        term->process(Eigen::VectorXd::Ones(1), f.inputs, f.outputs);
        near(position ? f.outputs.q_des[f.sid(0)] : f.outputs.v_des[f.sid(0)], .5);
    }
    std::cout << "Omnisteering action checks passed\n";
}
