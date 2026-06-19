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

using SensorSpec = std::variant<ContactSensorSpec, HeightScanSpec>;

}

#endif // __XBOT2_DEPLOY_POLICY_SENSOR_H
