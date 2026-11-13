#include "deploy_onnx.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace XBot::policy;

PYBIND11_MODULE(_bindings, m)
{
    m.doc() = "ONNX policy deployment runtime";

    py::class_<RobotInfo>(m, "RobotInfo")
        .def(py::init([] {
            RobotInfo robot{};
            robot.base_T_imu.setIdentity();
            return robot;
        }))
        .def_readwrite("joint_names", &RobotInfo::joint_names)
        .def_readwrite("joint_pos_min", &RobotInfo::joint_pos_min)
        .def_readwrite("joint_pos_max", &RobotInfo::joint_pos_max)
        .def_readwrite("joint_vel_max", &RobotInfo::joint_vel_max)
        .def_readwrite("body_names", &RobotInfo::body_names)
        .def_property("base_T_imu",
            [](const RobotInfo& robot) -> Eigen::Matrix4d { return robot.base_T_imu.matrix(); },
            [](RobotInfo& robot, const Eigen::Matrix4d& matrix) {
                robot.base_T_imu = Eigen::Affine3d(matrix);
            })
        .def_property("body_pose_in_base",
            [](const RobotInfo& robot) -> py::object {
                if(!robot.body_pose_in_base) return py::none();
                return py::cpp_function([callback = robot.body_pose_in_base](const std::string& name)
                    -> Eigen::Matrix4d { return callback(name).matrix(); });
            },
            [](RobotInfo& robot, std::function<Eigen::Matrix4d(const std::string&)> callback) {
                if(!callback) { robot.body_pose_in_base = {}; return; }
                robot.body_pose_in_base = [callback = std::move(callback)](const std::string& name) {
                    return Eigen::Affine3d(callback(name));
                };
            });

    py::class_<PolicyInfo>(m, "PolicyInfo")
        .def(py::init<>())
        .def_readwrite("control_dt", &PolicyInfo::control_dt)
        .def_readwrite("joint_names", &PolicyInfo::joint_names)
        .def_readwrite("stiffness", &PolicyInfo::stiffness)
        .def_readwrite("damping", &PolicyInfo::damping)
        .def_readwrite("joint_default_pos", &PolicyInfo::joint_default_pos)
        .def_readwrite("joint_default_vel", &PolicyInfo::joint_default_vel)
        .def_readwrite("joint_id_policy_to_robot", &PolicyInfo::joint_id_policy_to_robot)
        .def_readwrite("joint_id_robot_to_policy", &PolicyInfo::joint_id_robot_to_policy)
        .def_readwrite("command_size", &PolicyInfo::command_size)
        .def_readwrite("action_size", &PolicyInfo::action_size)
        .def_readwrite("height_scan_size", &PolicyInfo::height_scan_size);

    py::class_<CommandSpec>(m, "CommandSpec")
        .def(py::init<>())
        .def_readwrite("name", &CommandSpec::name)
        .def_readwrite("class_type", &CommandSpec::class_type)
        .def_readwrite("field_names", &CommandSpec::field_names)
        .def_readwrite("min", &CommandSpec::min)
        .def_readwrite("max", &CommandSpec::max)
        .def_readwrite("default_value", &CommandSpec::default_value)
        .def_readwrite("filter_cutoff_hz", &CommandSpec::filter_cutoff_hz);

    py::class_<SensorSpecBase>(m, "SensorSpecBase")
        .def(py::init<>())
        .def_readwrite("name", &SensorSpecBase::name)
        .def_readwrite("class_type", &SensorSpecBase::class_type);

    py::class_<ContactSensorSpec, SensorSpecBase>(m, "ContactSensorSpec")
        .def(py::init<>())
        .def_readwrite("num_contacts", &ContactSensorSpec::num_contacts);

    py::class_<HeightScanSpec, SensorSpecBase>(m, "HeightScanSpec")
        .def(py::init<>())
        .def_readwrite("size", &HeightScanSpec::size);

    py::class_<Inputs>(m, "Inputs")
        .def(py::init([] {
            Inputs inputs{};
            inputs.w_R_imu.setIdentity();
            inputs.imu_omega.setZero();
            inputs.imu_acc.setZero();
            return inputs;
        }))
        .def_readwrite("step", &Inputs::step)
        .def_readwrite("time_sec", &Inputs::time_sec)
        .def_readwrite("command", &Inputs::command)
        .def_readwrite("last_action", &Inputs::last_action)
        .def_readwrite("q", &Inputs::q)
        .def_readwrite("v", &Inputs::v)
        .def_readwrite("tau", &Inputs::tau)
        .def_readwrite("q_ref", &Inputs::q_ref)
        .def_readwrite("v_ref", &Inputs::v_ref)
        .def_readwrite("tau_ref", &Inputs::tau_ref)
        .def_readwrite("k", &Inputs::k)
        .def_readwrite("d", &Inputs::d)
        .def_readwrite("imu_omega", &Inputs::imu_omega)
        .def_readwrite("imu_acc", &Inputs::imu_acc)
        .def_readwrite("height_scan", &Inputs::height_scan)
        .def_readwrite("adaptation_parameters", &Inputs::adaptation_parameters)
        // Eigen quaternion coefficients use [x, y, z, w] ordering.
        .def_property("w_R_imu",
            [](const Inputs& inputs) -> Eigen::Vector4d { return inputs.w_R_imu.coeffs(); },
            [](Inputs& inputs, const Eigen::Vector4d& xyzw) {
                if(!xyzw.allFinite() || xyzw.norm() == 0.0)
                    throw py::value_error("w_R_imu must be a finite, nonzero quaternion in [x, y, z, w] order");
                inputs.w_R_imu.coeffs() = xyzw.normalized();
            });

    py::class_<Outputs>(m, "Outputs")
        .def(py::init<int, int>(), py::arg("action_size"), py::arg("num_joints"))
        .def_readwrite("raw_observation", &Outputs::raw_observation)
        .def_readwrite("raw_recurrent_states", &Outputs::raw_recurrent_states)
        .def_readwrite("raw_action", &Outputs::raw_action)
        .def_readwrite("q_des", &Outputs::q_des)
        .def_readwrite("v_des", &Outputs::v_des)
        .def_readwrite("tau_des", &Outputs::tau_des)
        .def_readwrite("k_des", &Outputs::k_des)
        .def_readwrite("d_des", &Outputs::d_des)
        .def_readwrite("ctrl_mode", &Outputs::ctrl_mode);

    py::class_<OnnxPolicy>(m, "OnnxPolicy")
        .def(py::init<std::string, std::string, std::string, RobotInfo, bool>(),
            py::arg("model_path"), py::arg("model_metadata_path"),
            py::arg("obs_group_override"), py::arg("robot_info"),
            py::arg("allow_missing_robot_joints") = false)
        .def("policyInfo", &OnnxPolicy::policyInfo)
        .def("command_specs", &OnnxPolicy::command_specs)
        .def("sensor_specs", &OnnxPolicy::sensor_specs)
        .def("recurrent_state_names", &OnnxPolicy::recurrent_state_names)
        .def("default_commands", &OnnxPolicy::default_commands)
        .def("sanitize_command", [](const OnnxPolicy& policy, const Inputs& inputs,
                                    const std::string& name, const Eigen::VectorXd& raw) {
            Eigen::VectorXd sanitized;
            std::string reason;
            const bool ok = policy.sanitize_command(inputs, name, raw, sanitized, &reason);
            return py::make_tuple(ok, sanitized, reason);
        }, py::arg("inputs"), py::arg("name"), py::arg("raw_command"))
        // Keep the GIL: action terms can invoke the Python body_pose_in_base callback.
        .def("run", &OnnxPolicy::run, py::arg("inputs"), py::arg("outputs"));
}
