#include "deploy_onnx.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <format>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

OnnxPolicy::OnnxPolicy(std::string model_path,
           std::string model_metadata_path, 
           std::string obs_group_override,
           RobotInfo robot_info)
    // Store paths/transforms up front; the ONNX session is opened lazily in init()
    // so construction stays cheap and errors happen when the caller explicitly starts the policy.
    : _model_path(std::move(model_path)),
      _model_metadata_path(std::move(model_metadata_path)),
      _base_T_imu(std::move(robot_info.base_T_imu)),
      // ONNX Runtime requires one environment object that outlives all sessions.
      _env(ORT_LOGGING_LEVEL_WARNING, "xbot2_deploy_policy"),
      // A null session lets init() replace it after all options have been configured.
      _session(nullptr),
      // Reused CPU memory info tells ORT that our std::vector buffers are host tensors.
      _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    init_onnxruntime();

    // Load the metadata file, which contains the policy info and observation/action configs.
    auto md = YAML::LoadFile(_model_metadata_path);

    // policy info to fill
    PolicyInfo policy_info;
    policy_info.joint_default_pos = md["default_joint_pos"].as<std::vector<double>>();
    if(md["default_joint_vel"])
    {
        policy_info.joint_default_vel = md["default_joint_vel"].as<std::vector<double>>();
    }
    else
    {
        policy_info.joint_default_vel.assign(policy_info.joint_default_pos.size(), 0.0);
    }
    policy_info.joint_names = md["joint_names"].as<std::vector<std::string>>();
    policy_info.action_size = _actionOutput().buffer.size();
    policy_info.joint_id_policy_to_robot.resize(policy_info.joint_names.size());
    policy_info.joint_id_robot_to_policy.assign(robot_info.joint_names.size(), -1);
    policy_info.control_dt = md["ctrl_dt"].as<double>();
    policy_info.stiffness = md["joint_stiffness"].as<std::vector<double>>();
    policy_info.damping = md["joint_damping"].as<std::vector<double>>();

    // map policy joint names to robot joint indices and vice versa
    std::unordered_map<std::string, int> robot_joint_ids;
    robot_joint_ids.reserve(robot_info.joint_names.size());
    for(std::size_t robot_id = 0; robot_id < robot_info.joint_names.size(); ++robot_id)
    {
        robot_joint_ids.emplace(robot_info.joint_names[robot_id], static_cast<int>(robot_id));
    }

    for(std::size_t policy_id = 0; policy_id < policy_info.joint_names.size(); ++policy_id)
    {
        const auto it = robot_joint_ids.find(policy_info.joint_names[policy_id]);
        if(it == robot_joint_ids.end())
        {
            throw std::runtime_error("Policy joint '" + policy_info.joint_names[policy_id] + "' not found in robot joint names");
        }

        const int robot_id = it->second;
        policy_info.joint_id_policy_to_robot[policy_id] = robot_id;
        policy_info.joint_id_robot_to_policy[robot_id] = static_cast<int>(policy_id);
    }

    // parse scene to get height scan size
    policy_info.height_scan_size = 0;
    if(auto scene = md["scene"]; scene && scene.IsMap())
    {
        for(auto pair : scene)
        {
            auto name = pair.first.as<std::string>();
            auto config = pair.second;
            if(!config.IsMap() || !config["class_type"])
            {
                continue;
            }

            auto class_type = config["class_type"].as<std::string>();

            if(class_type == "kyon_isaac.sensors.ray_caster:KyonRayCaster")
            {
                const double pattern_resolution = config["pattern_cfg"]["resolution"].as<double>();
                auto pattern_size = config["pattern_cfg"]["size"].as<std::pair<double, double>>();

                policy_info.height_scan_size = (static_cast<int>(std::round(pattern_size.first / pattern_resolution)) + 1) *
                                                (static_cast<int>(std::round(pattern_size.second / pattern_resolution)) + 1);

                _sensor_specs.push_back(HeightScanSpec{name, class_type, policy_info.height_scan_size});

                std::cout << std::format("[HeightScan] '{}' class: {} size: {}\n", name, class_type, policy_info.height_scan_size);
            }

            if(class_type == "isaaclab.sensors.camera.tiled_camera:TiledCamera")
            {
                DepthSpec spec;
                spec.name = name;
                spec.class_type = class_type;
                spec.height = config["height"].as<int>();
                spec.width = config["width"].as<int>();
                // near/far come from the depth obs term params, attached once groups are parsed
                _sensor_specs.push_back(spec);

                std::cout << std::format("[Depth] '{}' class: {} {}x{}\n",
                                         name, class_type, spec.width, spec.height);
            }
        }
    }

    // parse commands config
    auto cmd_cfg = md["commands"];
    for(auto pair : cmd_cfg) {
        auto name = pair.first.as<std::string>();
        auto config = pair.second;
        auto class_type = config["class_type"].as<std::string>();

        auto command_term = CommandTerm::create(name, class_type, robot_info, policy_info, config);
        policy_info.command_size[name] = command_term->size();
        _command_specs.push_back(command_term->spec());
        std::cout << "Command: " << name
                  << " class: " << class_type
                  << " size: " << command_term->size()
                  << std::endl;
        _command_terms.push_back(std::move(command_term));
    }

    // parse observation configs
    std::string obs_group = "policy";
    
    if(auto n = md["default_obs_group"])
    {
        obs_group = n.as<std::string>();
    }

    if(!obs_group_override.empty())
    {
        obs_group = obs_group_override;
    }

    // A CNN actor reads several groups: the 1D ones feed the flat "obs" input, each 2D one becomes
    // its own image input named after the group. deploy_export.py writes both lists; bundles without
    // them keep using default_obs_group above.
    if(auto n1d = md["actor_obs_groups_1d"]; n1d && n1d.size() > 0)
    {
        if(n1d.size() > 1)
        {
            throw std::runtime_error("Multiple 1D observation groups are not supported yet");
        }
        obs_group = n1d[0].as<std::string>();
    }

    if(auto n2d = md["actor_obs_groups_2d"])
    {
        for(auto g : n2d)
        {
            const auto group = g.as<std::string>();
            auto group_cfg = md["observations"][group];
            if(!group_cfg)
            {
                throw std::runtime_error("2D observation group '" + group + "' not found in metadata");
            }

            for(auto pair : group_cfg)
            {
                const auto cfg = pair.second["cfg"];
                if(!cfg.IsMap() || !cfg["params"] || !cfg["params"]["sensor_cfg"])
                {
                    continue;
                }

                const auto sensor_name = cfg["params"]["sensor_cfg"]["name"].as<std::string>();

                for(auto& s : _sensor_specs)
                {
                    auto* d = std::get_if<DepthSpec>(&s);
                    if(!d || d->name != sensor_name)
                    {
                        continue;
                    }

                    d->obs_group = group;
                    d->near = cfg["params"]["near"].as<double>(0.0);
                    d->far = cfg["params"]["far"].as<double>(1.0);

                    std::cout << std::format("[Depth] '{}' -> onnx input '{}', window [{}, {}] m\n",
                                             d->name, d->obs_group, d->near, d->far);
                }
            }
        }
    }

    auto obs_cfg = md["observations"][obs_group];
    int obs_size = 0;

    std::cout << std::format("Parsing observation group '{}' with {} terms\n", obs_group, obs_cfg.size());

    for(auto pair : obs_cfg) {
        
        const auto name = pair.first.as<std::string>();
        const auto config = pair.second["cfg"];

        if(config.IsScalar())
        {
            continue;
        }

        auto n_func = config["func"];

        if(!n_func)
        {
            continue;
        }

        auto func = n_func.as<std::string>();
        auto obs_term = ObsTerm::create(func, robot_info, policy_info, config);
        std::cout << "Observation: " << name << " func: " << func << " size: " << obs_term->size() << std::endl;
        
        _obs_terms.push_back(std::move(obs_term));
        _obs_term_names.push_back(name);  // TEMP DEBUG
        obs_size += _obs_terms.back()->size();

    }

    // Drop sensors no observation term reads. md["scene"] describes the whole training env, so a
    // depth bundle still lists the height scanner that only the critic consumed; subscribing to it
    // would make fill_inputs wait forever for data nothing needs.
    {
        std::set<std::string> used_sensors;

        auto collect = [&used_sensors](YAML::Node group_cfg) {
            if(!group_cfg)
            {
                return;
            }
            for(auto pair : group_cfg)
            {
                const auto cfg = pair.second["cfg"];
                if(cfg.IsMap() && cfg["params"] && cfg["params"]["sensor_cfg"])
                {
                    used_sensors.insert(cfg["params"]["sensor_cfg"]["name"].as<std::string>());
                }
            }
        };

        collect(obs_cfg);

        if(auto n2d = md["actor_obs_groups_2d"])
        {
            for(auto g : n2d)
            {
                collect(md["observations"][g.as<std::string>()]);
            }
        }

        std::erase_if(_sensor_specs, [&used_sensors](const SensorSpec& spec) {
            const auto& name = std::visit(
                [](const auto& s) -> const std::string& { return s.name; }, spec);

            if(used_sensors.count(name))
            {
                return false;
            }

            std::cout << std::format("[Sensor] '{}' is in the scene but no observation term reads it; skipping\n", name);
            return true;
        });
    }

    if(obs_size != _input_tensors.front().buffer.size())
    {
        throw std::runtime_error(
            std::format("Total observation size from metadata does not match model input size ({} != {})",
                        obs_size, _input_tensors.front().buffer.size()));   
    }

    // parse action configs
    auto action_cfg = md["actions"];
    int action_size = 0;
    for(auto pair : action_cfg) {
        auto name = pair.first.as<std::string>();
        auto config = pair.second["cfg"];
        auto func = config["class_type"].as<std::string>();

        auto action_term = ActionTerm::create(func, robot_info, policy_info, config);
        action_size += action_term->size();
        std::cout << "Action: " << name << " func: " << func << " size: " << action_term->size() << std::endl;
        _action_terms.push_back(std::move(action_term));
    }

    if(action_size != policy_info.action_size)
    {
        throw std::runtime_error(
            std::format("Total action size from metadata does not match model output size ({} != {})",
                        action_size, policy_info.action_size));   
    }

    std::cout << std::format("ctrl_dt: {} action_size: {} obs_size: {} (group: {}) \n", 
        policy_info.control_dt, action_size, obs_size, obs_group) << std::endl;

    _policy_info = std::move(policy_info);
}

PolicyInfo OnnxPolicy::policyInfo() const
{
    return _policy_info;
}

const std::vector<CommandSpec>& OnnxPolicy::command_specs() const
{
    return _command_specs;
}

const std::vector<SensorSpec> &OnnxPolicy::sensor_specs() const
{
    return _sensor_specs;
}

std::map<std::string, Eigen::VectorXd> OnnxPolicy::default_commands() const
{
    std::map<std::string, Eigen::VectorXd> commands;
    for(const auto& command_term : _command_terms)
    {
        commands[command_term->spec().name] = command_term->spec().default_value;
    }

    return commands;
}

bool OnnxPolicy::sanitize_command(const Inputs& inputs, 
                                  const std::string& name,
                                  const Eigen::VectorXd& raw_command,
                                  Eigen::VectorXd& sanitized_command,
                                  std::string* reason) const
{
    for(const auto& command_term : _command_terms)
    {
        if(command_term->spec().name == name)
        {
            return command_term->sanitize(inputs, raw_command, sanitized_command, reason);
        }
    }

    if(reason)
    {
        *reason = "Unknown command: " + name;
    }

    return false;
}

bool OnnxPolicy::run(const Inputs& inputs, Outputs& outputs)
{
    // Convert robot/control state into the ONNX input layout. This is intentionally
    // isolated because the exact packing will come from the metadata file.
    _fillInputBuffers(inputs);
    // Packing may resize or rename buffers once metadata support lands, so rebuild
    // the raw ORT name arrays immediately before execution.
    _refreshNamePointers();

    // Wrap each reusable std::vector as an Ort::Value without copying; ORT reads the
    // backing storage for the duration of the Run() call.
    std::vector<Ort::Value> input_values;
    input_values.reserve(_input_tensors.size());
    for (auto& tensor : _input_tensors) {
        input_values.push_back(Ort::Value::CreateTensor<float>(
            _memory_info,
            tensor.buffer.data(),
            tensor.buffer.size(),
            tensor.shape.data(),
            tensor.shape.size()));
    }

    // Execute the graph once using every discovered input and requesting every output.
    auto output_values = _session.Run(
        Ort::RunOptions{nullptr},
        _input_name_ptrs.data(),
        input_values.data(),
        input_values.size(),
        _output_name_ptrs.data(),
        _output_name_ptrs.size());

    // A count mismatch means the model metadata and returned ORT values disagree;
    // fail early before copying into the wrong output slot.
    if (output_values.size() != _output_tensors.size()) {
        throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
    }

    // Copy ORT-owned output tensors into reusable buffers. Dynamic output shapes are
    // taken from the actual value returned by this run.
    for (std::size_t i = 0; i < output_values.size(); ++i) {
        const auto output_info = output_values[i].GetTensorTypeAndShapeInfo();
        _output_tensors[i].shape = output_info.GetShape();
        _output_tensors[i].buffer.resize(_elementCount(_output_tensors[i].shape));

        auto* output_data = output_values[i].GetTensorMutableData<float>();
        std::copy_n(output_data, _output_tensors[i].buffer.size(), _output_tensors[i].buffer.begin());
    }

    // Feed h_out back in as h_in for the next step. ONNX graphs are stateless, so without this the
    // recurrence is dead and the policy acts on a permanently-zero hidden state -- silently.
    _carryRecurrentState();

    // Convert raw policy outputs into the typed Eigen output fields expected by the caller.
    _fillOutputs(outputs);

    return true;
}

void OnnxPolicy::init_onnxruntime()
{
    // Use single-threaded operator execution to reduce scheduling jitter in control loops.
    _session_options.SetIntraOpNumThreads(1);
    // Enable safe graph optimizations once at load time so each run() call does less work.
    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Loading the model here validates the ONNX file and exposes the graph I/O metadata.
    _session = Ort::Session(_env, _model_path.c_str(), _session_options);

    // Reinitialization should rebuild the buffers from the current model instead of
    // appending stale tensor descriptors from a previous session.
    _input_tensors.clear();
    _output_tensors.clear();
    _input_name_ptrs.clear();
    _output_name_ptrs.clear();

    // ORT owns temporary name allocations through this allocator; we immediately copy
    // the names into TensorBuffer::name so the public buffers have stable strings.
    Ort::AllocatorWithDefaultOptions allocator;

    // Discover every model input and allocate a matching reusable float buffer.
    const auto input_count = _session.GetInputCount();
    _input_tensors.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = _session.GetInputNameAllocated(i, allocator);
        _input_tensors.push_back(_makeTensorBuffer(name.get(), _session.GetInputTypeInfo(i)));
        _printTensorInfo("input", i, _input_tensors.back());
    }

    // Discover outputs too so run() can request all graph outputs and copy them back
    // into stable buffers that later metadata code can unpack.
    const auto output_count = _session.GetOutputCount();
    _output_tensors.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = _session.GetOutputNameAllocated(i, allocator);
        _output_tensors.push_back(_makeTensorBuffer(name.get(), _session.GetOutputTypeInfo(i)));
        _printTensorInfo("output", i, _output_tensors.back());
    }

    // ORT Run() takes raw const char* arrays; refresh them after vector growth is done
    // so the pointers reference the final string storage.
    _refreshNamePointers();
    _initialized = true;
}

OnnxPolicy::TensorBuffer OnnxPolicy::_makeTensorBuffer(const char* name, const Ort::TypeInfo& type_info)
{
    // The wrapper currently operates on float32 policy tensors; accepting other types
    // would require explicit conversion rules in the metadata packing layer.
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto element_type = tensor_info.GetElementType();
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            std::string("Unsupported ONNX tensor type for '") + name + "': expected float32, got " +
            _tensorElementTypeName(element_type));
    }

    // Store both the model-declared shape and the concrete runtime shape. Dynamic
    // dimensions are concretized for initial allocation and can be resized later.
    TensorBuffer tensor;
    tensor.name = name;
    tensor.element_type = element_type;
    tensor.model_shape = tensor_info.GetShape();
    tensor.shape = _concreteShape(tensor.model_shape);
    tensor.buffer.resize(_elementCount(tensor.shape));
    return tensor;
}

void OnnxPolicy::_printTensorInfo(const char* role, std::size_t index, const TensorBuffer& tensor)
{
    // Print the graph contract at startup so mismatches between metadata and ONNX
    // tensors are visible before the first policy run.
    std::cout << "xbot2_deploy_policy: ONNX " << role << "[" << index << "]"
              << " name='" << tensor.name << "'"
              << " type=" << _tensorElementTypeName(tensor.element_type)
              << " model_shape=" << _shapeToString(tensor.model_shape)
              << " concrete_shape=" << _shapeToString(tensor.shape)
              << " elements=" << tensor.buffer.size()
              << std::endl;
}

std::string OnnxPolicy::_shapeToString(const std::vector<int64_t>& shape)
{
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

const char* OnnxPolicy::_tensorElementTypeName(ONNXTensorElementDataType type)
{
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            return "undefined";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        default:
            return "unknown";
    }
}

std::vector<int64_t> OnnxPolicy::_concreteShape(const std::vector<int64_t>& model_shape)
{
    // ONNX uses negative/zero-like placeholders for symbolic dimensions; defaulting
    // them to one gives a valid buffer until metadata provides the real size.
    std::vector<int64_t> shape = model_shape;
    for (auto& dim : shape) {
        if (dim <= 0) {
            dim = 1;
        }
    }
    return shape;
}

std::size_t OnnxPolicy::_elementCount(const std::vector<int64_t>& shape)
{
    // Scalar tensors have an empty shape but still contain one value.
    if (shape.empty()) {
        return 1;
    }

    // Product of dimensions is the number of float elements required by ORT.
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        int64_t{1},
        [](int64_t lhs, int64_t rhs) {
            if (rhs <= 0) {
                throw std::runtime_error("ONNX tensor shape contains a non-positive concrete dimension");
            }
            return lhs * rhs;
        }));
}

void OnnxPolicy::_refreshNamePointers()
{
    // ORT Run() wants C arrays of names, while this class owns std::string names in
    // TensorBuffer. Rebuild the pointer arrays whenever buffers are recreated.
    _input_name_ptrs.clear();
    _output_name_ptrs.clear();

    _input_name_ptrs.reserve(_input_tensors.size());
    for (const auto& tensor : _input_tensors) {
        _input_name_ptrs.push_back(tensor.name.c_str());
    }

    _output_name_ptrs.reserve(_output_tensors.size());
    for (const auto& tensor : _output_tensors) {
        _output_name_ptrs.push_back(tensor.name.c_str());
    }
}

void OnnxPolicy::_fillInputBuffers(const Inputs& inputs)
{
    // Inputs are matched by NAME: an image input is named after its observation group, the
    // recurrent state carries over from the previous run, and what remains is the flat observation
    // vector. A feed-forward policy has exactly one input and lands on the last branch.
    TensorBuffer* flat = nullptr;

    for(auto& t : _input_tensors)
    {
        if(_isRecurrentInput(t.name))
        {
            continue;  // already holds h_out from the previous step, zeros on the first
        }

        const DepthSpec* depth = nullptr;
        for(const auto& s : _sensor_specs)
        {
            const auto* d = std::get_if<DepthSpec>(&s);
            if(d && d->obs_group == t.name)
            {
                depth = d;
                break;
            }
        }

        if(depth)
        {
            const auto it = inputs.depth.find(depth->obs_group);
            if(it != inputs.depth.end() && it->second.size() == t.buffer.size())
            {
                std::copy(it->second.begin(), it->second.end(), t.buffer.begin());
            }
            continue;
        }

        if(flat)
        {
            throw std::runtime_error("More than one non-image ONNX input: '" + flat->name + "' and '" + t.name + "'");
        }
        flat = &t;
    }

    if(!flat)
    {
        throw std::runtime_error("No flat observation input found on the ONNX model");
    }

    auto& tensor = *flat;
    int buffer_offset = 0;
    for(auto&& obs_term : _obs_terms)
    {
        Eigen::VectorXd term_output;
        obs_term->process(inputs, term_output);

        //std::cout << "Obs term output: " << term_output.transpose().format(2) << std::endl;

        std::copy_n(term_output.data(), term_output.size(), tensor.buffer.data() + buffer_offset);
        buffer_offset += term_output.size();
    }

    auto policy_input = Eigen::VectorXf::Map(tensor.buffer.data(), tensor.buffer.size());

    // --- TEMP DEBUG: every observation term with its name, throttled to ~1 Hz ---
    static int _dbg = 0;
    if(_dbg++ % 50 == 0)
    {
        std::cout << "[OBS] --------------------------------------------------" << std::endl;
        int off = 0;
        for(std::size_t t = 0; t < _obs_terms.size(); ++t)
        {
            const int n = _obs_terms[t]->size();
            std::cout << "[OBS] " << _obs_term_names[t] << " (" << n << "): ";
            for(int k = 0; k < n; ++k)
            {
                std::cout << policy_input(off + k) << " ";
            }
            std::cout << std::endl;
            off += n;
        }
    }
    // --- END TEMP DEBUG ---
}

void OnnxPolicy::_fillOutputs(Outputs& outputs)
{
    const auto& action_tensor = _actionOutput();
    auto policy_output = Eigen::VectorXf::Map(action_tensor.buffer.data(), action_tensor.buffer.size());
    //std::cout << "Policy output: " << policy_output.transpose().format(2) << std::endl;

    // set the raw action vector to the output, which is used as the last_action input in the next run
    outputs.raw_action = policy_output.cast<double>();

    // process each action term to fill the corresponding robot outputs
    const auto& tensor = action_tensor;
    int buffer_offset = 0;
    for(auto&& action_term : _action_terms)
    {
        auto raw_action = Eigen::VectorXf::Map(tensor.buffer.data() + buffer_offset, action_term->size());
        buffer_offset += action_term->size();

        //std::cout << "Raw action term: " << raw_action.transpose().format(2) << std::endl;

        action_term->process(raw_action.cast<double>(), outputs);
    }
}


bool OnnxPolicy::_isRecurrentInput(const std::string& name) const
{
    return name == "h_in" || name == "c_in";
}

void OnnxPolicy::_carryRecurrentState()
{
    // rsl-rl names these h_in/h_out, plus c_in/c_out for an LSTM
    for(auto& in : _input_tensors)
    {
        if(!_isRecurrentInput(in.name))
        {
            continue;
        }

        const std::string out_name = (in.name == "h_in") ? "h_out" : "c_out";

        for(const auto& out : _output_tensors)
        {
            if(out.name == out_name && out.buffer.size() == in.buffer.size())
            {
                std::copy(out.buffer.begin(), out.buffer.end(), in.buffer.begin());
                break;
            }
        }
    }
}


const OnnxPolicy::TensorBuffer& OnnxPolicy::_actionOutput() const
{
    // everything that is not recurrent state is the action head; rsl-rl exports exactly one
    for(const auto& out : _output_tensors)
    {
        if(out.name != "h_out" && out.name != "c_out")
        {
            return out;
        }
    }

    throw std::runtime_error("ONNX model has no non-recurrent output to read actions from");
}

}
