#include "deploy_onnx.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <format>

#include <yaml-cpp/yaml.h>

namespace XBot::policy {

OnnxPolicy::OnnxPolicy(std::string model_path,
           std::string model_metadata_path, 
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

    auto md = YAML::LoadFile(_model_metadata_path);

    // policy info to fill
    PolicyInfo policy_info;
    policy_info.joint_default_pos = md["default_joint_pos"].as<std::vector<double>>();
    policy_info.joint_names = md["joint_names"].as<std::vector<std::string>>();
    policy_info.action_size = _output_tensors.back().buffer.size();
    policy_info.joint_id_policy_to_robot.resize(policy_info.joint_names.size());
    policy_info.joint_id_robot_to_policy.assign(robot_info.joint_names.size(), -1);
    policy_info.control_dt = md["ctrl_dt"].as<double>();
    policy_info.stiffness = md["joint_stiffness"].as<std::vector<double>>();
    policy_info.damping = md["joint_damping"].as<std::vector<double>>();

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

    // parse commands config
    auto cmd_cfg = md["commands"];
    for(auto pair : cmd_cfg) {
        auto name = pair.first.as<std::string>();
        auto config = pair.second;
        auto ranges = config["ranges"];
        policy_info.command_size[name] = ranges.size();
    }

    // parse observation configs
    auto obs_cfg = md["observations"]["policy"];
    int obs_size = 0;

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
        obs_size += _obs_terms.back()->size();

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

    std::cout << "Action size: " << policy_info.action_size << " Observation size: " << obs_size << std::endl;

    _policy_info = std::move(policy_info);
}

PolicyInfo OnnxPolicy::policyInfo() const
{
    return _policy_info;
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
    auto& tensor = _input_tensors.back();
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
    //std::cout << "Policy input: " << policy_input.transpose().format(2) << std::endl;
}

void OnnxPolicy::_fillOutputs(Outputs& outputs)
{
    auto policy_output = Eigen::VectorXf::Map(_output_tensors.back().buffer.data(), _output_tensors.back().buffer.size());
    //std::cout << "Policy output: " << policy_output.transpose().format(2) << std::endl;

    // set the raw action vector to the output, which is used as the last_action input in the next run
    outputs.raw_action = policy_output.cast<double>();

    // process each action term to fill the corresponding robot outputs
    auto& tensor = _output_tensors.back();
    int buffer_offset = 0;
    for(auto&& action_term : _action_terms)
    {
        auto raw_action = Eigen::VectorXf::Map(tensor.buffer.data() + buffer_offset, action_term->size());
        buffer_offset += action_term->size();

        //std::cout << "Raw action term: " << raw_action.transpose().format(2) << std::endl;

        action_term->process(raw_action.cast<double>(), outputs);
    }
}

}
