#include "parameters.hpp"

#include <yaml-cpp/yaml.h>

namespace mrf {

void Parameters::fromConfig(const std::string& file_name) {
    const YAML::Node cfg{YAML::LoadFile(file_name)};

    std::string tmp;

    getParam(cfg, "ks", ks);
    getParam(cfg, "kd", kd);
    getParam(cfg, "kn", kn);
    getParam(cfg, "discontinuity_threshold", discontinuity_threshold);
    getParam(cfg, "smoothness_rate", smoothness_rate);
    getParam(cfg, "max_num_iterations", solver.max_num_iterations);
    getParam(cfg, "radius_normal_estimation", radius_normal_estimation);
    getParam(cfg, "neighbor_search", neighbor_search);

    getParam(cfg, "estimate_normals", estimate_normals);
    getParam(cfg, "use_functor_normal_distance", use_functor_normal_distance);
    getParam(cfg, "use_functor_smoothness_normal", use_functor_smoothness_normal);
    getParam(cfg, "use_functor_normal", use_functor_normal);
    getParam(cfg, "use_functor_distance", use_functor_distance);
    getParam(cfg, "use_functor_smoothness_distance", use_functor_smoothness_distance);

    getParam(cfg, "pin_normals", pin_normals);
    getParam(cfg, "pin_distances", pin_distances);

    getParam(cfg, "estimate_covariances", estimate_covariances);

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
        } else if (tmp == "mean_depth") {
            initialization = Initialization::mean_depth;
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

    getParam(cfg, "loss_function_scale", loss_function_scale);
    if (getParam(cfg, "loss_function", tmp)) {
        if (tmp == "trivial") {
            loss_function = std::make_shared<ceres::TrivialLoss>();
        } else if (tmp == "huber") {
            loss_function = std::make_shared<ceres::HuberLoss>(loss_function_scale);
        } else {
            LOG(WARNING) << "No parameter " << tmp << " available.";
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
    return os << "discontinuity threshold: " << p.discontinuity_threshold << std::endl
              << "ks: " << p.ks << std::endl
              << "kd: " << p.kd << std::endl
              << "kn: " << p.kn << std::endl
              << "max_iterations: " << p.solver.max_num_iterations << std::endl
              << "limits: " << static_cast<int>(p.limits) << std::endl
              << "smoothness_rate: " << p.smoothness_rate << std::endl
              << "radius_normal_estimation: " << p.radius_normal_estimation << std::endl
              << "custom_depth_limit_min: " << p.custom_depth_limit_min << std::endl
              << "custom_depth_limit_max: " << p.custom_depth_limit_max << std::endl
              << "neighbor_search: " << p.neighbor_search << std::endl
              << "estimate_normals: " << p.estimate_normals << std::endl
              << "use_functor_normal_distance: " << p.use_functor_normal_distance << std::endl
              << "use_functor_smoothness_normal: " << p.use_functor_smoothness_normal << std::endl
              << "use_functor_normal: " << p.use_functor_normal << std::endl
              << "use_functor_distance: " << p.use_functor_distance << std::endl
              << "use_functor_smoothness_distance: " << p.use_functor_smoothness_distance
              << std::endl
              << "pin_normals: " << p.pin_normals << std::endl
              << "pin_distances: " << p.pin_distances << std::endl
              << "loss_function_scale: " << p.loss_function_scale << std::endl
              << "initialization: " << static_cast<int>(p.initialization) << std::endl
              << "neighborhood: " << static_cast<int>(p.neighborhood) << std::endl
              << "estimate_covariances: " << p.estimate_covariances << std::endl;
}
}
