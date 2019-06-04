//
// Created by mikezhu on 6/3/19.
//
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera_model_pinhole.h"
#include "downsample.hpp"
//#include "evaluate.hpp"
#include "io.hpp"
//#include "noise.hpp"
//#include "quality.hpp"
#include "solver.hpp"


/** @brief Creates map of command line options
 *  @param argc Number of arguments
 *  @param argv Arguments
 *  @return Map of the options */
bool ParseCommandLine(int argc, char **argv,
                      boost::program_options::variables_map *vm) {
  boost::program_options::options_description desc("Supported Parameters");
  desc.add_options() ("help", "Produce help message")
      ("input",
       boost::program_options::value<std::string>()->required(),
       "Path to input folders")
      ("output",
       boost::program_options::value<std::string>()->required(),
       "Path to log output")
      ("parameters",
       boost::program_options::value<std::string>()->required(),
       "Path to parameters file");
  try {
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), *vm);
    if (vm->count("help")) {
      std::cerr << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }
    boost::program_options::notify(*vm);
  } catch (std::exception& e) {
    std::cerr << "Error" << e.what() << std::endl;
    std::cerr << desc << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown error!" << std::endl;
    return false;
  }
  return true;
}


int main(int argc, char** argv) {
  google::InitGoogleLogging("eval_kitti");
  google::InstallFailureSignalHandler();

  boost::program_options::variables_map boost_args;
  if (!ParseCommandLine(argc, argv, &boost_args)) {
    std::cerr << "Failed to parse the command" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const std::string input = boost_args["input"].as<std::string>();
  const std::string output_path = boost_args["output"].as<std::string>();
  const std::string param_filepath = boost_args["parameters"].as<std::string>();

  std::string photo_path = input + "/208253600.png";
  std::string pcd_path = input + "/208253600.pcd";

  mrf::Parameters p_mrf(param_filepath);
  /// none(only init)
  // p_mrf.use_functor_distance = false;
  // p_mrf.use_functor_normal = false;
  // p_mrf.use_functor_normal_distance = false;
  // p_mrf.use_functor_smoothness_distance = false;
  /// normal
  // p_mrf.use_functor_distance = true;
  // p_mrf.use_functor_normal = false;
  // p_mrf.use_functor_normal_distance = true;
  // p_mrf.use_functor_smoothness_distance = false;
  /// Diebel
  // p_mrf.use_functor_distance = true;
  // p_mrf.use_functor_normal = false;
  // p_mrf.use_functor_normal_distance = false;
  // p_mrf.use_functor_smoothness_distance = true;

  // TODO: start evaluation process
  boost::filesystem::path output_data_path{output_path + "/data/"};
  boost::filesystem::create_directories(output_data_path);

  LOG(INFO) << "Photo path: " << photo_path;
  LOG(INFO) << "Loading and converting images";
  cv::Mat img_photo{cv::imread(photo_path, cv::IMREAD_COLOR)};

  LOG(INFO) << "Depth photo: " << img_photo.depth();              // CV_8U
  LOG(INFO) << "Channels photo: " << img_photo.channels();

  /**
   * \attention Units in dataset are in mm! Need to convert to meters
   */
  //img_depth.convertTo(img_depth, cv::DataType<double>::type, 0.001);
  img_photo.convertTo(img_photo, cv::DataType<float>::type);

  LOG(INFO) << "Export converted images";
  mrf::exportImage(img_photo, output_data_path.string() + "rgb_", true);

  const size_t rows = img_photo.rows;
  const size_t cols = img_photo.cols;

  LOG(INFO) << "Create camera model";
  //TODO: use real camera model info
  int imgWidth = img_photo.cols;   // 1226
  int imgHeight = img_photo.rows;  // 370
  double focalLength = 707.091187;
  double principalPointU = 601.887329;  // cols / 2,
  double principalPointV = 183.110397;  // rows / 2

  auto cam = std::make_shared<CameraModelPinhole>(imgWidth,
                                                  imgHeight,
                                                  focalLength,
                                                  principalPointU,
                                                  principalPointV);

  LOG(INFO) << "Create cloud from depth image";
  using KittiPoint = pcl::PointXYZI;
  using KittiCloud = pcl::PointCloud<KittiPoint>;

  using PclPoint = pcl::PointXYZINormal;
  using PclCloud = pcl::PointCloud<PclPoint>;

  using PointOut = pcl::PointXYZRGBNormal;

  KittiCloud::Ptr rawCloud(new KittiCloud);
  // load the file
  if (pcl::io::loadPCDFile<KittiPoint>(pcd_path, *rawCloud) == -1) {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return EXIT_FAILURE;
  }

  PclCloud::Ptr inputCloud(new PclCloud);
  inputCloud = mrf::estimateNormals<KittiPoint, PclPoint>(rawCloud,
      p_mrf.radius_normal_estimation, false);

//  LOG(INFO) << "Downsample cloud";
//  PclCloud::Ptr inputCloud_downsampled;

  cv::Mat image_in = img_photo;

  const mrf::Data<PclPoint> in{inputCloud, image_in};

  LOG(INFO) << "Call MRF library";
  mrf::Data<PointOut> out;
  mrf::Solver mrf_solver(cam, p_mrf);
  const mrf::ResultInfo result_info{mrf_solver.solve(in, out)};

  LOG(INFO) << "Export output data";
  mrf::exportImage(out.image, output_data_path.string() + "out_", true, false, true);
  mrf::exportCloud<PointOut>(out.cloud, output_data_path.string() + "out_");

  LOG(INFO) << "Export result info";
  exportResultInfo(result_info, output_data_path.string() + "result_info_");
  return EXIT_SUCCESS;
}
