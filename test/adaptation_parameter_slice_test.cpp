#include "obs_term.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition)
{
    if(!condition) throw std::runtime_error("Adaptation parameter slice test failed");
}

auto make_term(int start, int end, int history = 0, bool clip = false)
{
    auto config = YAML::Load("scale: null\nclip: null\nhistory_length: 0\nparams: {}\n");
    config["params"]["start"] = start;
    config["params"]["end"] = end;
    config["history_length"] = history;
    if(clip) config["clip"] = std::vector<double>{-0.5, 0.5};
    return XBot::policy::ObsTerm::create(
        "kyon_isaac.tasks.locomotion.velocity.mdp.observations:adaptation_parameter_slice",
        XBot::policy::RobotInfo{}, XBot::policy::PolicyInfo{}, config);
}

void check_slices(const XBot::policy::Inputs& inputs)
{
    for(const auto& bounds : {std::pair{0, 1}, std::pair{1, 4}, std::pair{4, 8}})
    {
        auto term = make_term(bounds.first, bounds.second);
        Eigen::VectorXd output;
        term->process(inputs, output);
        require(term->size() == bounds.second - bounds.first);
        require(output.isApprox(inputs.adaptation_parameters.segment(bounds.first, term->size())));
    }
}

} // namespace

int main(int argc, char** argv)
{
    XBot::policy::Inputs inputs;
    Eigen::Matrix<double, 8, 1> defaults;
    defaults << 0., 0., 0., 0., 1., 1., 1., 1.;
    require(inputs.adaptation_parameters == defaults);
    check_slices(inputs);
    inputs.adaptation_parameters << 2.5, 0.01, -0.02, 0.03, 0.4, 0.6, 0.8, 1.2;
    check_slices(inputs);

    // Exercise configs directly from an exported metadata file when supplied.
    if(argc > 1)
    {
        auto terms = YAML::LoadFile(argv[1])["observations"]["policy"];
        for(const auto& name : {"payload_mass", "payload_com", "foot_friction"})
        {
            auto config = terms[name]["cfg"];
            auto term = XBot::policy::ObsTerm::create(config["func"].as<std::string>(),
                XBot::policy::RobotInfo{}, XBot::policy::PolicyInfo{}, config);
            Eigen::VectorXd output;
            term->process(inputs, output);
            require(output.isApprox(inputs.adaptation_parameters.segment(
                config["params"]["start"].as<int>(), term->size())));
        }
    }

    for(const auto& bounds : {std::pair{-1, 1}, std::pair{0, 9}, std::pair{4, 4}, std::pair{4, 1}})
    {
        bool threw = false;
        try { make_term(bounds.first, bounds.second); }
        catch(const std::runtime_error&) { threw = true; }
        require(threw);
    }

    auto history = make_term(0, 1, 2, true);
    Eigen::VectorXd output;
    history->process(inputs, output);
    require(history->size() == 2 && output.isApprox(Eigen::Vector2d(0.5, 0.5)));
    inputs.adaptation_parameters(0) = -2.;
    history->process(inputs, output);
    require(output.isApprox(Eigen::Vector2d(0.5, -0.5)));

    inputs.adaptation_parameters(7) = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try { make_term(4, 8)->process(inputs, output); }
    catch(const std::runtime_error&) { threw = true; }
    require(threw);
}
