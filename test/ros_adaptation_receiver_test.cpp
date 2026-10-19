#include "ros_adaptation_receiver.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
using Message = std_msgs::msg::Float64MultiArray;

void check(bool condition, const char* reason)
{
    if(!condition) throw std::runtime_error(reason);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<rclcpp::Node>("adaptation_receiver_test",
            rclcpp::NodeOptions().use_intra_process_comms(true));
        policy_deploy::RosAdaptationReceiver receiver(*node);
        auto publisher = node->create_publisher<Message>("~/adaptation_parameters", 1);
        Message current;
        int received = 0;
        auto subscriber = node->create_subscription<Message>("~/adaptation_parameters/current", 1,
            [&](Message::ConstSharedPtr msg) { current = *msg; ++received; });
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        auto spin = [&] {
            const auto deadline = std::chrono::steady_clock::now() + 50ms;
            do {
                executor.spin_some();
                std::this_thread::sleep_for(1ms);
            } while(std::chrono::steady_clock::now() < deadline);
        };
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while(publisher->get_subscription_count() == 0 || subscriber->get_publisher_count() == 0)
        {
            check(std::chrono::steady_clock::now() < deadline, "topic discovery failed");
            spin();
        }
        XBot::policy::Inputs inputs;
        const std::vector<double> defaults{0, 0, 0, 0, 1, 1, 1, 1};
        auto read = [&](const std::vector<double>& expected) {
            const int previous = received;
            receiver.fill_inputs(inputs);
            spin();
            check(received > previous, "current values were not published");
            check(current.data == expected, "published values differ from expected snapshot");
            for(int i = 0; i < 8; ++i)
                check(inputs.adaptation_parameters(i) == expected[i], "Inputs differ from publication");
        };
        read(defaults);
        Message valid;
        valid.data = {-2.5, 0.01, -0.02, 0.03, 0, 0.9, 1, 0.7};
        publisher->publish(valid);
        spin();
        read(valid.data);
        for(int error = 0; error < 5; ++error)
        {
            auto invalid = valid;
            if(error == 0) invalid.data.pop_back();
            if(error == 1) invalid.data.push_back(1);
            if(error == 2) invalid.data[0] = std::numeric_limits<double>::quiet_NaN();
            if(error == 3) invalid.data[2] = std::numeric_limits<double>::infinity();
            if(error == 4) invalid.data[7] = -0.1;
            publisher->publish(invalid);
            spin();
            read(valid.data);
        }
        // Elapsed policy time and changes to the caller's Inputs do not expire the estimate.
        inputs.time_sec = 1e9;
        inputs.adaptation_parameters.setZero();
        read(valid.data);
        valid.data = defaults;
        publisher->publish(valid);
        spin();
        read(defaults);
        std::cout << "Adaptation subscription, publication, validation and retention tests passed\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
}
