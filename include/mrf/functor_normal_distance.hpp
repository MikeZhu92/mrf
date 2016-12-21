#pragma once

#include <memory>
#include <ostream>
#include <ceres/autodiff_cost_function.h>
#include <Eigen/src/Geometry/Hyperplane.h>
#include <Eigen/src/Geometry/ParametrizedLine.h>

namespace mrf {

struct FunctorNormalDistance {

    using Ptr = std::shared_ptr<FunctorNormalDistance>;

    static constexpr size_t DimDepth = 1;
    static constexpr size_t DimResidual = 1;
    static constexpr size_t DimNormal = 3;

    inline FunctorNormalDistance(const double& w, const Eigen::Vector3d& support_this,
                                 const Eigen::Vector3d& direction_this,
                                 const Eigen::Vector3d& support_nn,
                                 const Eigen::Vector3d& direction_nn)
            : w_{w}, support_this_{support_this}, direction_this_{direction_this},
              support_nn_{support_this}, direction_nn_{direction_this} {};

    template <typename T>
    inline bool operator()(const T* const depth_this, const T* const depth_nn,
                           const T* const normal, T* res) const {
        using namespace Eigen;
        const ParametrizedLine<T, 3> ray_nn(support_nn_.cast<T>(), direction_nn_.cast<T>());
        Hyperplane<T, 3> plane_this(Map<const Vector3<T>>(normal, DimNormal),
                                    support_this_.cast<T>() +
                                        direction_this_.cast<T>() * depth_this[0]);
        res[0] =
            T(w_) * (ray_nn.intersectionPoint(plane_this) - ray_nn.pointAt(depth_nn[0])).norm();
        return true;
    }

    inline static ceres::CostFunction* create(const double& w, const Eigen::Vector3d& support_this,
                                              const Eigen::Vector3d& direction_this,
                                              const Eigen::Vector3d& support_nn,
                                              const Eigen::Vector3d& direction_nn) {
        return new ceres::AutoDiffCostFunction<FunctorNormalDistance, DimResidual, DimDepth,
                                               DimDepth, DimNormal>(
            new FunctorNormalDistance(w, support_this, direction_this, support_nn, direction_nn));
    }

    inline friend std::ostream& operator<<(std::ostream& os, const FunctorNormalDistance& f) {
        return os << "Weight: " << f.w_ << std::endl;
    }

    inline void setWeight(const double w) {
        w_ = w;
    }

private:
    double w_; ///< Weight

    Eigen::Vector3d support_this_, support_nn_;
    Eigen::Vector3d direction_this_, direction_nn_;
};
}
