// #include <pcl/test/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/features/feature.h>
#include <pcl/features/cgf.h>
#include <pcl/io/pcd_io.h>

using namespace pcl;
using namespace pcl::io;

using KdTreePtr = search::KdTree<PointXYZ>::Ptr;

// PointCloud<PointXYZ> cloud;
using PointInT = PointXYZ;
using PointOutT = CGFSignature16;
static PointCloud<PointInT>::Ptr cloud (new PointCloud<PointInT> ());
Indices indices;
KdTreePtr tree;


// TEST (PCL, CGFEstimation)
int test()
{
  std::cout << "\ntesting\n";
  CGFEstimation<PointInT, PointOutT> cgf_estimation = CGFEstimation<PointInT, PointOutT> (2,2,2);
  cgf_estimation.setInputCloud (cloud);
  PointCloud<PointOutT>::Ptr feature_cloud (new PointCloud<PointOutT> ());
  
  // cgf_estimation.compute (*feature_cloud);
  cgf_estimation.setCompression("/home/zoe/catkin_ws/src/pcl/test/features/NeuralNetwork_test.csv");

  Eigen::VectorXf input;
  input.setOnes(16);
  std::cout << "input: " << input;
  Eigen::VectorXf output(16);
  cgf_estimation.getCompression().applyNN(input, output);

  std::cout << "output: " << output << "\n";
  // int sz = cgf_estimation.getCompression().layers_[0].inputSize();
  // std::cout << sz;
  // cgf_estimation.getCompression().
  return 0;
}


int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "No test file given. Please download `bun0.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }

  if (loadPCDFile<PointXYZ> (argv[1], *cloud) < 0)
  {
    std::cerr << "Failed to read test file. Please download `bun0.pcd` and pass its path to the test." << std::endl;
    return (-1);
  }

  indices.resize (cloud->size ());
  for (std::size_t i = 0; i < indices.size (); ++i)
    indices[i] = static_cast<int> (i);

  tree.reset (new search::KdTree<PointXYZ> (false));
  tree->setInputCloud (cloud->makeShared ());

  return test();
  // testing::InitGoogleTest (&argc, argv);
  // return (RUN_ALL_TESTS ());
}