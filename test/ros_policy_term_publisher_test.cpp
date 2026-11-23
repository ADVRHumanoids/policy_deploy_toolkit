#include "ros_policy_term_publisher.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
using Message = std_msgs::msg::Float32MultiArray;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<rclcpp::Node>("term_publisher_test",
            rclcpp::NodeOptions().use_intra_process_comms(true));
        policy_deploy::RosPolicyTermPublisher observations(*node, "~/policy/observations",
            {{"gravity", 0, 3}, {"joint_history", 3, 6}});
        policy_deploy::RosPolicyTermPublisher actions(*node, "~/policy/actions",
            {{"legs", 0, 2}, {"wheels", 2, 1}});
        const std::vector<std::string> topics{
            "~/policy/observations/gravity", "~/policy/observations/joint_history",
            "~/policy/actions/legs", "~/policy/actions/wheels"};
        std::vector<Message> received(topics.size());
        std::vector<int> counts(topics.size(), 0);
        std::vector<rclcpp::Subscription<Message>::SharedPtr> subscriptions;
        for(std::size_t i = 0; i < topics.size(); ++i)
        {
            subscriptions.push_back(node->create_subscription<Message>(topics[i], 10,
                [&, i](Message::ConstSharedPtr msg) { received[i] = *msg; ++counts[i]; }));
        }
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        Eigen::VectorXf obs(9);
        obs << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        Eigen::VectorXd action(3);
        action << -1.5, 2.5, -3.5;
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        for(const auto& subscription : subscriptions)
        {
            while(subscription->get_publisher_count() == 0)
            {
                if(std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Topic discovery failed");
                executor.spin_some();
                std::this_thread::sleep_for(1ms);
            }
        }
        for(int tick = 0; tick < 2; ++tick)
        {
            observations.publish(obs);
            actions.publish(action);
            const auto tick_deadline = std::chrono::steady_clock::now() + 3s;
            while(counts[0] <= tick || counts[1] <= tick || counts[2] <= tick || counts[3] <= tick)
            {
                if(std::chrono::steady_clock::now() >= tick_deadline)
                    throw std::runtime_error("Term publication timed out");
                executor.spin_some();
                std::this_thread::sleep_for(1ms);
            }
            const float delta = static_cast<float>(tick * 10);
            const std::vector<std::vector<float>> expected{
                {1 + delta, 2 + delta, 3 + delta},
                {4 + delta, 5 + delta, 6 + delta, 7 + delta, 8 + delta, 9 + delta},
                {-1.5f + delta, 2.5f + delta}, {-3.5f + delta}};
            for(std::size_t i = 0; i < topics.size(); ++i)
            {
                if(received[i].data != expected[i] || counts[i] != tick + 1)
                    throw std::runtime_error("Incorrect term slice or publication count: " + topics[i]);
            }
            obs.array() += 10;
            action.array() += 10;
        }
        std::cout << "Observation history and action topic slices passed\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
}
