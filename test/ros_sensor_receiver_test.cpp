#include "ros_sensor_receiver.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace policy_deploy {

struct RosSensorReceiverTest {
    using Receiver = RosSensorReceiver;
    using Cloud = sensor_msgs::msg::PointCloud2;

    static void check(bool condition, const char* message)
    {
        if(!condition) throw std::runtime_error(message);
    }

    static rclcpp::Time time(double seconds)
    {
        return rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
    }

    static Eigen::Quaterniond rpy(double roll, double pitch, double yaw)
    {
        return Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
    }

    static void add_scan(Receiver& receiver, int size)
    {
        Receiver::HeightScanState state;
        state.spec = XBot::policy::HeightScanSpec{"scan", "test", size};
        state.value = Eigen::VectorXd::Zero(size);
        state.stamp = receiver._node.now();
        receiver._height_scan_states.emplace("scan", std::move(state));
    }

    // Mixed types and unused per-point/row padding deliberately exercise offsets.
    static Cloud cloud(const std::vector<Eigen::Vector3d>& points, double seconds,
                       const std::string& frame = "base_link", unsigned width = 0)
    {
        Cloud msg;
        msg.header.frame_id = frame;
        msg.header.stamp = time(seconds);
        msg.width = width ? width : points.size();
        msg.height = points.size() / msg.width;
        msg.point_step = 32;
        msg.row_step = msg.width * msg.point_step + 8;
        msg.data.resize(msg.height * msg.row_step);
        for(int axis = 0; axis < 3; ++axis)
        {
            sensor_msgs::msg::PointField field;
            field.name = std::string(1, "xyz"[axis]);
            field.offset = axis * 8;
            field.datatype = axis == 1 ? field.FLOAT32 : field.FLOAT64;
            field.count = 1;
            msg.fields.push_back(field);
        }
        for(std::size_t i = 0; i < points.size(); ++i)
        {
            const auto offset = (i / msg.width) * msg.row_step + (i % msg.width) * msg.point_step;
            for(int axis = 0; axis < 3; ++axis)
            {
                if(axis == 1)
                {
                    const float value = points[i](axis);
                    std::memcpy(msg.data.data() + offset + axis * 8, &value, sizeof(value));
                }
                else
                {
                    const double value = points[i](axis);
                    std::memcpy(msg.data.data() + offset + axis * 8, &value, sizeof(value));
                }
            }
        }
        return msg;
    }

    static void set_tf(Receiver& receiver, const std::string& source, const Eigen::Affine3d& pose)
    {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.frame_id = "base_link";
        tf.child_frame_id = source;
        tf.header.stamp = time(999); // Static TF must also work for clouds at t=1.
        const Eigen::Quaterniond q(pose.linear());
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        check(receiver._tf_buffer.setTransform(tf, "test", true), "insert static TF");
    }

    static Eigen::VectorXd read(Receiver& receiver)
    {
        XBot::policy::Inputs inputs;
        check(receiver.fill_inputs(inputs, receiver._node.now()), "scan should be ready");
        return inputs.height_scan;
    }

    static void transforms(rclcpp::Node& node)
    {
        const std::vector<Eigen::Quaterniond> attitudes = {
            rpy(0, 0, 0), rpy(0, 0, 1.2), rpy(0.4, 0, 0),
            rpy(0, -0.3, 0), rpy(0.4, -0.3, 2.0)};
        const std::vector<Eigen::Vector3d> expected = {
            {0.3, 0.2, -0.8}, {-0.4, 0.1, -0.8}, {0.2, -0.2, -0.8}, {0.0, 0.1, -0.5}};
        const auto mounting = rpy(0.2, -0.4, 0.3);
        Eigen::Affine3d extrinsics = Eigen::Affine3d::Identity();
        extrinsics.linear() = rpy(-0.2, 0.1, 0.6).toRotationMatrix();
        extrinsics.translation() = Eigen::Vector3d(0.4, -0.2, 0.3);
        for(const auto& attitude : attitudes)
        {
            Receiver receiver(node, 0.5, mounting.toRotationMatrix());
            add_scan(receiver, 4);
            receiver.update_imu(attitude * mounting);
            set_tf(receiver, "lidar", extrinsics);
            const Eigen::Matrix3d R = attitude.toRotationMatrix();
            const double yaw = std::atan2(R(1, 0), R(0, 0));
            const Eigen::Matrix3d horizontal_R_base = rpy(0, 0, -yaw).toRotationMatrix() * R;
            std::vector<Eigen::Vector3d> source_points;
            for(const auto& point : expected)
                source_points.push_back(extrinsics.inverse() * (horizontal_R_base.transpose() * point));
            receiver.store_height_scan("scan", cloud(source_points, 1, "lidar", 2));
            const auto values = read(receiver);
            for(int i = 0; i < values.size(); ++i)
                check(std::abs(values(i) - expected[i].z()) < 1e-7, "leveled height/order mismatch");
            // Once cached, changes in TF must not change the static extrinsics.
            set_tf(receiver, "lidar", Eigen::Affine3d::Identity());
            receiver.store_height_scan("scan", cloud(source_points, 1, "lidar", 2));
            check((read(receiver) - values).norm() < 1e-12, "static TF should be cached");
            // A cloud already in the horizontal frame is unchanged with consistent TF.
            Eigen::Affine3d base_T_horizontal = Eigen::Affine3d::Identity();
            base_T_horizontal.linear() = horizontal_R_base.transpose();
            set_tf(receiver, "horizontal", base_T_horizontal);
            receiver.store_height_scan("scan", cloud(expected, 1, "horizontal", 2));
            check((read(receiver) - values).norm() < 1e-7, "leveled cloud corrected twice");
        }
    }

    static void latest_imu(rclcpp::Node& node)
    {
        Receiver receiver(node, 0.5, Eigen::Matrix3d::Identity());
        add_scan(receiver, 1);
        receiver.update_imu(Eigen::Quaterniond::Identity());
        receiver.update_imu(rpy(0.4, 0, 1.2));
        // A non-unit quaternion is normalized; invalid updates retain the latest valid one.
        Eigen::Quaterniond scaled = rpy(0.4, 0, 1.2);
        scaled.coeffs() *= -2;
        receiver.update_imu(scaled);
        receiver.update_imu(Eigen::Quaterniond(0, 0, 0, 0));
        receiver.update_imu(Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(), 0, 0, 0));
        const double expected = std::sin(0.4) * 2 - std::cos(0.4);
        for(double stamp : {0.0, 1.0, 999.0})
        {
            receiver.store_height_scan("scan", cloud({{1, 2, -1}}, stamp));
            check(std::abs(read(receiver)(0) - expected) < 1e-12,
                  "cloud must use latest valid IMU regardless of timestamp");
        }
        receiver.update_imu(Eigen::Quaterniond::Identity());
        check(std::abs(read(receiver)(0) - expected) < 1e-12, "old scan was retransformed");
        receiver.store_height_scan("scan", cloud({{1, 2, -1}}, 0));
        check(std::abs(read(receiver)(0) + 1) < 1e-12, "new scan did not use new IMU reading");
    }

    static void pending_and_validation(rclcpp::Node& node)
    {
        Receiver receiver(node, 0.5, Eigen::Matrix3d::Identity());
        add_scan(receiver, 1);
        receiver.store_height_scan("scan", cloud({{1, 2, -1}}, 1.2));
        XBot::policy::Inputs inputs;
        check(!receiver.fill_inputs(inputs, node.now()), "scan should wait for initial IMU");
        receiver.store_height_scan("scan", cloud({{1, 2, -2}}, 1.2));
        receiver.update_imu(Eigen::Quaterniond::Identity());
        check(std::abs(read(receiver)(0) + 2) < 1e-12, "newest pending scan not used");
        auto valid = cloud({{1, 2, -3}}, 1.4);
        for(int error = 0; error < 11; ++error)
        {
            auto bad = valid;
            switch(error)
            {
                case 0: bad.header.frame_id.clear(); break;
                case 1: bad.is_bigendian = true; break;
                case 2: bad.fields.pop_back(); break;
                case 3: bad.fields[0].datatype = sensor_msgs::msg::PointField::INT32; break;
                case 4: bad.fields[1].count = 0; break;
                case 5: bad.fields[2].offset = 31; break;
                case 6: bad.point_step = 0; break;
                case 7: bad.row_step = 1; break;
                case 8: bad.data.resize(23); break;
                case 9: bad.width = 2; break;
                case 10: bad = cloud({{std::numeric_limits<double>::infinity(), 2, -3}}, 1.4); break;
            }
            receiver.store_height_scan("scan", bad);
            check(std::abs(read(receiver)(0) + 2) < 1e-12, "invalid cloud replaced valid scan");
        }
        // Uniform FLOAT32 and FLOAT64 clouds must work as well as mixed fields.
        for(const auto datatype : {sensor_msgs::msg::PointField::FLOAT32,
                                   sensor_msgs::msg::PointField::FLOAT64})
        {
            auto uniform = valid;
            for(int axis = 0; axis < 3; ++axis)
            {
                uniform.fields[axis].datatype = datatype;
                const double value = axis == 0 ? 1 : (axis == 1 ? 2 : -2);
                const float float_value = value;
                if(datatype == sensor_msgs::msg::PointField::FLOAT32)
                    std::memcpy(uniform.data.data() + axis * 8, &float_value, sizeof(float_value));
                else
                    std::memcpy(uniform.data.data() + axis * 8, &value, sizeof(value));
            }
            receiver.store_height_scan("scan", uniform);
            check(std::abs(read(receiver)(0) + 2) < 1e-12, "uniform XYZ type failed");
        }
        receiver.store_height_scan("scan", cloud({{1, 2, -4}}, 1.4, "missing"));
        const auto start = std::chrono::steady_clock::now();
        check(std::abs(read(receiver)(0) + 2) < 1e-12, "missing TF replaced valid scan");
        check(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100), "TF lookup blocked");
        set_tf(receiver, "missing", Eigen::Affine3d::Identity());
        check(std::abs(read(receiver)(0) + 4) < 1e-12, "pending scan did not recover after TF arrival");
        receiver.store_height_scan("scan", cloud({{1, 2, -5}}, 1.6, "not_yet_available"));
        auto& state = receiver._height_scan_states.at("scan");
        check(state.pending.has_value(), "pending expiration test not queued");
        state.pending->received = node.now() - rclcpp::Duration::from_seconds(1);
        check(std::abs(read(receiver)(0) + 4) < 1e-12 && !state.pending, "expired scan not discarded");
        state.stamp = node.now() - rclcpp::Duration::from_seconds(1);
        check(std::abs(read(receiver)(0) + 4) < 1e-12, "stale scan was not reused");

        Receiver immediate(node, 0, Eigen::Matrix3d::Identity());
        add_scan(immediate, 1);
        immediate.update_imu(Eigen::Quaterniond::Identity());
        immediate.store_height_scan("scan", cloud({{0, 0, -1}}, 1));
        check(std::abs(read(immediate)(0) + 1) < 1e-12, "zero timeout with IMU/TF available failed");
        immediate.store_height_scan("scan", cloud({{0, 0, -2}}, 1.1));
        check(std::abs(read(immediate)(0) + 2) < 1e-12, "zero timeout must use latest IMU for any timestamp");
        immediate.store_height_scan("scan", cloud({{0, 0, -2}}, 1, "absent"));
        check(!immediate._height_scan_states.at("scan").pending, "zero timeout must not wait for TF");
        check(std::abs(read(immediate)(0) + 2) < 1e-12, "missing TF replaced last valid scan");
    }

    static void listener(rclcpp::Node& node)
    {
        Receiver receiver(node, 2.0, Eigen::Matrix3d::Identity());
        add_scan(receiver, 1);
        receiver.update_imu(Eigen::Quaterniond::Identity());
        tf2_ros::StaticTransformBroadcaster broadcaster(&node);
        geometry_msgs::msg::TransformStamped tf;
        tf.header.frame_id = "base_link";
        tf.child_frame_id = "receiver_test_sensor";
        tf.header.stamp = time(999);
        tf.transform.rotation.w = 1;
        tf.transform.translation.z = 0.25;
        broadcaster.sendTransform(tf);
        receiver.store_height_scan("scan", cloud({{0, 0, -1}}, 1, tf.child_frame_id));
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node.get_node_base_interface());
        XBot::policy::Inputs inputs;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        bool ready = false;
        while(std::chrono::steady_clock::now() < deadline)
        {
            executor.spin_some();
            if(receiver.fill_inputs(inputs, node.now()))
            {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        executor.remove_node(node.get_node_base_interface());
        check(ready, "listener did not receive static TF on node executor");
        check(std::abs(inputs.height_scan(0) + 0.75) < 1e-12, "received static TF translation incorrect");
    }

    static void run(rclcpp::Node& node)
    {
        transforms(node);
        latest_imu(node);
        pending_and_validation(node);
        listener(node);
    }
};
} // namespace policy_deploy

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<rclcpp::Node>("ros_sensor_receiver_test");
        policy_deploy::RosSensorReceiverTest::run(*node);
        std::cout << "Height scan transform, latest IMU, TF, and validation tests passed\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
}
