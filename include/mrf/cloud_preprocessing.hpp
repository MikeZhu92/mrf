#pragma once

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>

namespace mrf {

template <typename T, typename U>
const typename pcl::PointCloud<T>::Ptr estimateNormals(const typename pcl::PointCloud<T>::ConstPtr& in,
                                              const double& radius) {
    using namespace pcl;
    const typename PointCloud<U>::Ptr out{new PointCloud<U>};
    NormalEstimationOMP<T, U> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(in);
    ne.compute(out);
    return out;
}

template <typename T>
const typename pcl::PointCloud<T>::Ptr transform(const typename pcl::PointCloud<T>::ConstPtr& in,
                                        const Eigen::Affine3d& tf) {
    using namespace pcl;
    const typename PointCloud<T>::Ptr out{new PointCloud<T>};
    transformPointCloud(*in, *out, tf);
    return out;
}
}