#include "parameters.hpp"

#include <yaml-cpp/yaml.h>

namespace mrf {

void Parameters::fromConfig(const std::string& file_name) {
    const YAML::Node cfg{YAML::LoadFile(file_name)};

    std::string tmp;

    getParam(cfg, "ks", ks);
    getParam(cfg, "kd", kd);
    getParam(cfg, "discontinuity_threshold", discontinuity_threshold);
    getParam(cfg, "max_iterations", max_iterations);
    getParam(cfg, "radius_normal_estimation", radius_normal_estimation);

    if (getParam(cfg, "limits", tmp)) {
        if (tmp == "none") {
            limits = Limits::none;
        } else if (tmp == "custom") {
            limits = Limits::custom;
        } else if (tmp == "adaptive") {
            limits = Limits::adaptive;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    getParam(cfg, "custom_depth_limit_min", custom_depth_limit_min);
    getParam(cfg, "custom_depth_limit_max", custom_depth_limit_max);

    if (getParam(cfg, "initialization", tmp)) {
        if (tmp == "none") {
            initialization = Initialization::none;
        } else if (tmp == "nearest_neighbor") {
            initialization = Initialization::nearest_neighbor;
        } else if (tmp == "triangles") {
            initialization = Initialization::triangles;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }

    if (getParam(cfg, "neighborhood", tmp)) {
        if (tmp == "four") {
            neighborhood = Neighborhood::four;
        } else if (tmp == "eight") {
            neighborhood = Neighborhood::eight;
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
}
}