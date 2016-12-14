#pragma once

#include <memory>
#include <pcl/point_cloud.h>
#include <Eigen/src/Geometry/Transform.h>
#include <opencv2/core/core.hpp>

namespace mrf {

template <typename T>
struct Data {
    using Ptr = std::shared_ptr<Data>;

    using Cloud = pcl::PointCloud<T>;
    using Image = cv::Mat;
    using Transform = Eigen::Affine3d;

    inline Data(const typename Cloud::Ptr& cl, const Image& img, const Transform& tf)
            : cloud{cl}, image{img}, transform{tf} {};

    inline static Ptr create(const typename Cloud::Ptr& cl, const cv::Mat& img,
                             const Transform& tf) {
        return std::make_shared<Data>(cl, img, tf);
    }

    inline friend std::ostream& operator<<(std::ostream& os, const Data& d) {
        os << "Image size: " << d.image.rows << " x " << d.image.cols << std::endl
           << "Number of cloud points: " << cloud->size() << std::endl
           << "Transform: \n"
           << transform.matrix();
        return os;
    }

    const typename Cloud::Ptr cloud;
    Image image;
    Transform transform; ///< Transform between camera to laser
};
}
