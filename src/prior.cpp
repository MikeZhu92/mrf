#include "prior.hpp"

#include <flann/flann.h>

#include "smoothness_weight.hpp"

namespace mrf {

using DataType = double;
using Point = pcl_ceres::Point<DataType>;
using PixelMapT = std::map<Pixel, Point, PixelLess>;
using RayMapT = std::map<Pixel, Eigen::ParametrizedLine<double, 3>, PixelLess>;
using treeT = std::unique_ptr<flann::Index<flann::L2_Simple<DataType>>>;
using EigenT = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DistanceType = flann::L2_Simple<DataType>;

flann::Matrix<DataType> convertEigen2FlannRow(const EigenT& mEigen) {
    flann::Matrix<DataType> mFlann(new DataType[mEigen.size()], mEigen.rows(), mEigen.cols());

    for (size_t n = 0; n < (unsigned)mEigen.size(); ++n) {
        *(mFlann.ptr() + n) = *(mEigen.data() + n);
    }
    return mFlann;
}

bool insideTriangle(const Pixel& p,
                    const Eigen::Vector2i& first,
                    const Eigen::Vector2i& second,
                    const Eigen::Vector2i& third) {
    const Eigen::Vector2i& P{p.col, p.row};
    const Eigen::Vector2i& AC{first - third};
    const Eigen::Vector2i& BA{second - first};
    const Eigen::Vector2i& CB{third - second};
    const Eigen::Vector2i& PA{P - first};
    const Eigen::Vector2i& PB{P - second};
    const Eigen::Vector2i& PC{P - third};

    const int u{(BA[0] * PA[1] - (BA[1] * PA[0]))};
    const int v{(BA[0] * (-1 * AC[1])) - (BA[1] * (-1 * AC[0]))};
    if (u * v >= 0) {
        if (v == 0) {
            return false;
        }
        const int u2{(AC[0] * PC[1] - (AC[1] * PC[0]))};
        const int v2{(AC[0] * (-1 * CB[1])) - (AC[1] * (-1 * CB[0]))};
        if (u2 * v2 >= 0) {
            if (v2 == 0) {
                return false;
            }
            const int u3{(CB[0] * PB[1] - (CB[1] * PB[0]))};
            const int v3{(CB[0] * (-1 * BA[1])) - (CB[1] * (-1 * BA[0]))};
            if (u3 * v3 >= 0) {
                if (v3 == 0) {
                    return false;
                }
                return true;
            }
        }
    }
    return false;
}

bool getTriangleNeighbours(const std::vector<int>& neighbours_in,
                           const Eigen::Matrix2Xi& coordinates,
                           const Pixel& p,
                           std::vector<int>& triangle_neighbours) {
    if (neighbours_in.size() < 3) {
        return false;
    }
    size_t i{0};
    while (i < (neighbours_in.size() - 2)) {
        const Eigen::Vector2i& first_coorinate{coordinates(0, neighbours_in[i]),
                                               coordinates(1, neighbours_in[i])};
        triangle_neighbours[0] = neighbours_in[i];
        size_t j{1 + i};
        while (j < (neighbours_in.size() - 1)) {
            const Eigen::Vector2i& second_coorinate{coordinates(0, neighbours_in[j]),
                                                    coordinates(1, neighbours_in[j])};
            triangle_neighbours[1] = neighbours_in[j];
            size_t n{j + 1};
            while (n < (neighbours_in.size())) {
                const Eigen::Vector2i& third_coorinate{coordinates(0, neighbours_in[n]),
                                                       coordinates(1, neighbours_in[n])};
                triangle_neighbours[2] = neighbours_in[n];
                if (insideTriangle(p, first_coorinate, second_coorinate, third_coorinate)) {
                    return true;
                }
                n++;
            }
            j++;
        }
        i++;
    }
    return false;
}

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates,
                               const treeT& tree,
                               const Pixel& p,
                               const int num_neigh) {
    DistanceType::ElementType queryData[] = {static_cast<DistanceType::ElementType>(p.col),
                                             static_cast<DistanceType::ElementType>(p.row)};

    const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
    std::vector<std::vector<int>> indices_vec;
    std::vector<std::vector<DataType>> dist_vec;
    tree->knnSearch(query, indices_vec, dist_vec, num_neigh, flann::SearchParams(32));
    return indices_vec[0];
}

std::vector<int> getNeighbours(const Eigen::Matrix2Xi& coordinates,
                               std::vector<DataType>& distances,
                               const treeT& tree,
                               const Pixel& p,
                               const int num_neigh) {
    DistanceType::ElementType queryData[] = {static_cast<DistanceType::ElementType>(p.col),
                                             static_cast<DistanceType::ElementType>(p.row)};

    const flann::Matrix<DistanceType::ElementType> query(queryData, 1, 2);
    std::vector<std::vector<int>> indices_vec;
    std::vector<std::vector<DataType>> dist_vec;
    tree->knnSearch(query, indices_vec, dist_vec, num_neigh, flann::SearchParams(32));
    distances = dist_vec[0];
    return indices_vec[0];
}


double pointIntersection(const Eigen::ParametrizedLine<double, 3>& ray,
                         const std::vector<Eigen::Vector3d>& neighbours) {

    const Eigen::Vector3d direction_1_norm{neighbours[0] - neighbours[1]}; //.normalized();
    const Eigen::Vector3d direction_2_norm{neighbours[2] - neighbours[1]}; //.normalized();

    const Eigen::Vector3d normal = (direction_1_norm.cross(direction_2_norm)).normalized();

    const Eigen::Hyperplane<double, 3> plane(normal, neighbours[0]);
    const Eigen::Vector3d p_int{ray.intersectionPoint(plane)};
    return (p_int - ray.origin()).norm();
}

void addSeedPoints(const RayMapT& rays,
                   const PixelMapT& projection,
                   Eigen::MatrixXd& depth_est,
                   Eigen::MatrixXd& certainty) {
    for (auto const& el : projection) {
        const Pixel& p{el.first};
        certainty(p.row, p.col) = 1;
        const Eigen::ParametrizedLine<double, 3>& ray{rays.at(p)};
        const Eigen::Hyperplane<double, 3> plane(ray.direction(), el.second.position);
        depth_est(p.row, p.col) =
            (rays.at(p).intersectionPoint(plane) - rays.at(p).origin()).norm();
    }
}

void estimatePrior(const RayMapT& rays,
                   const PixelMapT& projection,
                   const size_t& rows,
                   const size_t& cols,
                   const Parameters& param,
                   Eigen::MatrixXd& depth_est,
                   Eigen::MatrixXd& certainty) {
    if (param.initialization == Parameters::Initialization::none) {
        LOG(INFO) << "Estimate prior via 'none' method";
        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }
    double max_depth{0};
    double min_depth{10000};
    double sum{0};
    size_t i{0};
    Eigen::Matrix2Xi coordinates(2, projection.size());
    for (auto const& el : projection) {
        coordinates(0, i) = el.first.col;
        coordinates(1, i++) = el.first.row;
        const Pixel& p{el.first};
        const Eigen::ParametrizedLine<double, 3>& ray{rays.at(p)};
        const Eigen::Hyperplane<double, 3> plane(ray.direction(), el.second.position);
        double val{(rays.at(p).intersectionPoint(plane) - rays.at(p).origin()).norm()};
        sum += val;
        max_depth = std::max(max_depth, val);
        min_depth = std::min(min_depth, val);
    }

    if (param.initialization == Parameters::Initialization::mean_depth) {
        LOG(INFO) << "Estimate prior via 'mean_depth' method";
        double mean_depth{sum / projection.size()};
        LOG(INFO) << "mean depth is " << mean_depth;
        depth_est = mean_depth * Eigen::MatrixXd::Ones(rows, cols);
        certainty = 0 * Eigen::MatrixXd::Ones(rows, cols);
        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }

    std::unique_ptr<flann::Index<DistanceType>> kd_index_ptr_;
    flann::Matrix<DistanceType::ElementType> flann_dataset{
        convertEigen2FlannRow(coordinates.transpose().cast<DataType>())};
    kd_index_ptr_ =
        std::make_unique<flann::Index<DistanceType>>(flann_dataset, flann::KDTreeIndexParams(8));
    kd_index_ptr_->buildIndex(flann_dataset);

    if (param.initialization == Parameters::Initialization::nearest_neighbor) {
        LOG(INFO) << "Estimate prior via 'nearest_neighbor' method";
        for (auto const& el : rays) {
            const Pixel& p{el.first};
            std::vector<int> neighbours{getNeighbours(coordinates, kd_index_ptr_, p, 1)};
            const Pixel nn(coordinates(0, neighbours[0]), coordinates(1, neighbours[0]));
            const Eigen::ParametrizedLine<double, 3>& ray{rays.at(nn)};
            const Eigen::Hyperplane<double, 3> plane(ray.direction(), projection.at(nn).position);
            depth_est(p.row, p.col) =
                (rays.at(nn).intersectionPoint(plane) - rays.at(nn).origin()).norm();
        }

        addSeedPoints(rays, projection, depth_est, certainty);
        return;
    }

    if (param.initialization == Parameters::Initialization::weighted_neighbor) {
        LOG(INFO) << "Estimate prior via 'weighted neighbors' method";
        for (auto const& el : rays) {
            const Pixel& p{el.first};
            std::vector<double> dist_neighbors;
            std::vector<int> all_neighbours{
                getNeighbours(coordinates, dist_neighbors, kd_index_ptr_, p, 4)};
            double w_sum{0};
            double d_sum{0};
            std::vector<double> w_vector;
            std::vector<Pixel> nn_vector;
            for (size_t n = 0; n < all_neighbours.size(); n++) {
                const Pixel nn(coordinates(0, all_neighbours[n]),
                               coordinates(1, all_neighbours[n]));
                nn_vector.emplace_back(nn);
                double w{smoothnessWeight(p,
                                          nn,
                                          param.discontinuity_threshold,
                                          param.smoothness_weight_min,
                                          param.smoothness_weighting,
                                          param.smoothness_rate)};
                w_sum += w;
                w_vector.emplace_back(w);
                d_sum += dist_neighbors[n];
            }
            double alpha{0}, beta{0};
            for (size_t n = 0; n < all_neighbours.size(); n++) {
                const Eigen::ParametrizedLine<double, 3>& ray{rays.at(nn_vector[n])};
                double depth_nn{(projection.at(nn_vector[n]).position - ray.origin()).norm()};
                alpha += (1 - dist_neighbors[n] / d_sum) / 3 * depth_nn;
                beta += w_vector[n] / w_sum * depth_nn;
            }
            depth_est(p.row, p.col) = (alpha + beta) / 2;
            const Pixel nn(coordinates(0, all_neighbours[0]), coordinates(1, all_neighbours[0]));
        }
        addSeedPoints(rays, projection, depth_est, certainty);
    }


    if (param.initialization == Parameters::Initialization::triangles) {
        LOG(INFO) << "Estimate prior via 'triangles' method";

        size_t inside_triangles{0};
        for (auto const& el : rays) {
            const Pixel& p{el.first};
            std::vector<int> all_neighbours{
                getNeighbours(coordinates, kd_index_ptr_, p, param.neighbor_search)};
            std::vector<int> triangle_neighbours(3);
            const bool found_triangle{
                getTriangleNeighbours(all_neighbours, coordinates, p, triangle_neighbours)};

            if (found_triangle) {
                inside_triangles++;
                std::vector<Pixel> neighbour_pixels;
                std::vector<Eigen::Vector3d> neighbour_points;
                std::vector<double> dists;
                double sum_dist{0};
                for (size_t i = 0; i < 3; i++) {
                    Pixel c(coordinates(0, triangle_neighbours[i]),
                            coordinates(1, triangle_neighbours[i]));
                    neighbour_pixels.emplace_back(c);
                    neighbour_points.emplace_back(projection.at(c).position);
                    dists.emplace_back(
                        std::sqrt(std::pow(p.row - c.row, 2) + std::pow(p.col - c.col, 2)));
                    sum_dist += dists.back();
                }
                depth_est(p.row, p.col) = pointIntersection(rays.at(p), neighbour_points);
                certainty(p.row, p.col) = 0.2;
            }
            if (!found_triangle || depth_est(p.row, p.col) > max_depth ||
                depth_est(p.row, p.col) < min_depth || depth_est(p.row, p.col) != depth_est(p.row, p.col) ||
                std::isinf(depth_est(p.row, p.col))) {
                Pixel nn(coordinates(0, all_neighbours[0]), coordinates(1, all_neighbours[0]));
                const Eigen::ParametrizedLine<double, 3>& ray{rays.at(nn)};
                const Eigen::Hyperplane<double, 3> plane(ray.direction(),
                                                         projection.at(nn).position);
                depth_est(p.row, p.col) =
                    (rays.at(nn).intersectionPoint(plane) - rays.at(nn).origin()).norm();
                certainty(p.row, p.col) = 0;
            }
        }
        LOG(INFO) << "Points with found triangle: " << inside_triangles
                  << ", in % of image: " << (double)inside_triangles / rays.size() * 100 << "%";
        addSeedPoints(rays, projection, depth_est, certainty);
    }
}
}
