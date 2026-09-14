#ifndef __XBOT2_DEPLOY_POLICY_SENSOR_H
#define __XBOT2_DEPLOY_POLICY_SENSOR_H

#include "types.h"

#include <string>
#include <variant>

namespace XBot::policy {

struct SensorSpecBase
{
    std::string name;
    std::string class_type;
};

struct ContactSensorSpec : public SensorSpecBase
{
    int num_contacts;
};

struct HeightScanSpec : public SensorSpecBase
{
    int size;
};

// A depth camera feeding its own 2D ONNX input, named after the observation group it belongs to
// (the group name is what rsl-rl uses as the input name when exporting a CNN model).
struct DepthSpec : public SensorSpecBase
{
    int height{0};
    int width{0};
    double near{0.0};   // normalization window, from the obs term params
    double far{1.0};
    std::string obs_group;  // ONNX input name
    int size() const { return height * width; }
};

using SensorSpec = std::variant<ContactSensorSpec, HeightScanSpec, DepthSpec>;

}

#endif // __XBOT2_DEPLOY_POLICY_SENSOR_H
