#ifndef XBOT2_DEPLOY_POLICY_DEPLOY_ONNX_H
#define XBOT2_DEPLOY_POLICY_DEPLOY_ONNX_H

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "types.h"
#include "obs_term.h"
#include "action_term.h"
#include "command_term.h"

namespace XBot::policy {

class OnnxPolicy {

public:

OnnxPolicy(std::string model_path,
           std::string model_metadata_path,
           RobotInfo robot_info);

PolicyInfo policyInfo() const;

const std::vector<CommandSpec>& command_specs() const;

std::map<std::string, Eigen::VectorXd> default_commands() const;

bool sanitize_command(const std::string& name,
                      const Eigen::VectorXd& raw_command,
                      Eigen::VectorXd& sanitized_command,
                      std::string* reason = nullptr) const;

bool run(const Inputs& inputs, Outputs& outputs);

private:

struct TensorBuffer {
    std::string name;
    ONNXTensorElementDataType element_type;
    std::vector<int64_t> model_shape;
    std::vector<int64_t> shape;
    std::vector<float> buffer;
};

void init_onnxruntime();

static TensorBuffer _makeTensorBuffer(const char* name, const Ort::TypeInfo& type_info);
static void _printTensorInfo(const char* role, std::size_t index, const TensorBuffer& tensor);
static std::string _shapeToString(const std::vector<int64_t>& shape);
static const char* _tensorElementTypeName(ONNXTensorElementDataType type);
static std::vector<int64_t> _concreteShape(const std::vector<int64_t>& model_shape);
static std::size_t _elementCount(const std::vector<int64_t>& shape);

void _refreshNamePointers();
void _fillInputBuffers(const Inputs& inputs);
void _fillOutputs(Outputs& outputs);

std::string _model_path;
std::string _model_metadata_path;
Eigen::Affine3d _base_T_imu;

Ort::Env _env;
Ort::SessionOptions _session_options;
Ort::Session _session;
Ort::MemoryInfo _memory_info;

std::vector<TensorBuffer> _input_tensors;
std::vector<TensorBuffer> _output_tensors;
std::vector<const char*> _input_name_ptrs;
std::vector<const char*> _output_name_ptrs;
bool _initialized{false};

std::vector<std::unique_ptr<ObsTerm>> _obs_terms;
std::vector<std::unique_ptr<ActionTerm>> _action_terms;
std::vector<std::unique_ptr<CommandTerm>> _command_terms;
std::vector<CommandSpec> _command_specs;
PolicyInfo _policy_info;

};

}

#endif // XBOT2_DEPLOY_POLICY_DEPLOY_ONNX_H
