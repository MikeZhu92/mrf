#include <limits>
#include <map>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <util_ceres/eigen_quaternion_parameterization.h>

#include "cloud_preprocessing.hpp"
#include "eigen.hpp"
#include "functor_distance.hpp"
#include "functor_smoothness.hpp"
#include "image_preprocessing.hpp"
#include "neighbors.hpp"

namespace mrf {

template <typename T>
bool Solver::solve(Data<T>& data) {

    LOG(INFO) << "Preprocess image";
    const cv::Mat img{gradientSobel(data.image)};

    LOG(INFO) << "Preprocess and transform cloud";
    using PointT = pcl::PointXYZINormal;
    const pcl::PointCloud<PointT>::Ptr cl{
        estimateNormals<T, PointT>(transform<T>(data.cloud, data.transform),
                                   params_.radius_normal_estimation)}; ///< \todo Check if
                                                                       /// transform is correct or
    /// needs to be inverted

    LOG(INFO) << "Compute point projection in camera image";
    const Eigen::Matrix3Xd pts_3d{cl->getMatrixXfMap().topRows<3>().cast<double>()};
    LOG(INFO) << "Rows: " << pts_3d.rows() << ", Cols: " << pts_3d.cols();

    Eigen::Matrix2Xd img_pts_raw{Eigen::Matrix2Xd::Zero(2, pts_3d.cols())};
    std::vector<bool> in_img{camera_->getImagePoints(pts_3d, img_pts_raw)};
    int width, height;
    camera_->getImageSize(width, height);
    const size_t dim{width * height};
    std::map<Eigen::Vector2d, Eigen::Vector3d, EigenLess> projection;
    for (size_t c = 0; c < in_img.size(); c++) {
        const int row = img_pts_raw(0, c);
        const int col = img_pts_raw(1, c);
        if (in_img[c] && (row > 0) && (row < height) && (col > 0) && (col < width)) {
            projection.insert(std::make_pair(img_pts_raw.col(c), pts_3d.col(c)));
        }
    }

    LOG(INFO) << "Create optimization problem";
    ceres::Problem problem(params_.problem);
    Eigen::MatrixXd depth_est{Eigen::MatrixXd::Zero(height, width)};
    std::vector<FunctorDistance::Ptr> functors_distance;
    functors_distance.reserve(projection.size());
    Eigen::Quaterniond rotation{data.transform.rotation()};
    Eigen::Vector3d translation{data.transform.translation()};
    problem.AddParameterBlock(rotation.coeffs().data(), 4,
                              new util_ceres::EigenQuaternionParameterization);
    problem.AddParameterBlock(translation.data(), 3);

    LOG(INFO) << "Add distance costs";
    for (auto const& el : projection) {
        functors_distance.emplace_back(FunctorDistance::create(el.second, params_.kd));
        problem.AddResidualBlock(
            functors_distance.back()->toCeres(), new ceres::HuberLoss(1),
            &depth_est(static_cast<int>(el.first[0]), static_cast<int>(el.first[1])),
            rotation.coeffs().data(), translation.data());
    }

    LOG(INFO) << "Add smoothness costs and depth limits";
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            for (auto const& n :
                 getNeighbors(Pixel(row, col), height, width, params_.neighborhood)) {
                problem.AddResidualBlock(
                    FunctorSmoothness::create(img.at<float>(col, row) * params_.ks), nullptr,
                    &depth_est(row, col), &depth_est(n.row, n.col));
            }
        }
    }

    LOG(INFO) << "Add depth limits";
    if (params_.use_custom_depth_limits) {
        for (size_t row = 0; row < height; row++) {
            for (size_t col = 0; col < width; col++) {
                problem.SetParameterLowerBound(&depth_est(row, col), 0,
                                               params_.custom_depth_limit_min);
                problem.SetParameterUpperBound(&depth_est(row, col), 0,
                                               params_.custom_depth_limit_max);
            }
        }
    } else {
        const double depth_max{pts_3d.colwise().norm().maxCoeff()};
        const double depth_min{pts_3d.colwise().norm().minCoeff()};
        LOG(INFO) << "Use adaptive limits. Min: " << depth_min << ", Max: " << depth_max;
        for (size_t row = 0; row < height; row++) {
            for (size_t col = 0; col < width; col++) {
                problem.SetParameterLowerBound(&depth_est(row, col), 0, depth_min);
                problem.SetParameterUpperBound(&depth_est(row, col), 0, depth_max);
            }
        }
    }

    LOG(INFO) << "Set parameterization";
    problem.SetParameterBlockConstant(rotation.coeffs().data());
    problem.SetParameterBlockConstant(translation.data());

    LOG(INFO) << "Check parameters";
    std::string err_str;
    if (params_.solver.IsValid(&err_str)) {
        LOG(INFO) << "Residuals set up correctly";
    } else {
        LOG(ERROR) << err_str;
    }

    LOG(INFO) << "Solve problem";
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solve(params_.solver, &problem, &summary);
    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
}
}
