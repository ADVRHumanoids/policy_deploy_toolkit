#ifndef __DEPLOY_ONNX_H__
#define __DEPLOY_ONNX_H__

#include <Eigen/Dense>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace XBot::policy {

class OnnxPolicy {

public:

typedef Eigen::Ref<const Eigen::VectorXd> VectorXdConstRef;

OnnxPolicy(std::string model_path,
           std::string model_metadata_path, 
           Eigen::Affine3d base_T_imu = Eigen::Affine3d::Identity())
{

}

void init()
{

}

struct Inputs {
    VectorXdConstRef q; 
    VectorXdConstRef v; 
    VectorXdConstRef tau;
    VectorXdConstRef q_ref;
    VectorXdConstRef v_ref;
    VectorXdConstRef tau_ref;
    VectorXdConstRef k;
    VectorXdConstRef d;
    Eigen::Quaterniond w_R_imu;
    Eigen::Vector3d imu_omega;
    Eigen::Vector3d imu_acc;
};

struct Outputs {
    Eigen::VectorXd q_des;
    Eigen::VectorXd v_des;
    Eigen::VectorXd tau_des;
    Eigen::VectorXd k_des;
    Eigen::VectorXd d_des;
};

bool run(const Inputs& inputs, Outputs& outputs)
{
    // TODO(alaurenzi) fill policy input from arguments

    // TODO(alaurenzi) run the policy and fill the output variables

    return true;
}

private:



};

}

#endif // __DEPLOY_ONNX_H__