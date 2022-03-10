// #include <pcl/test/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/features/feature.h>
#include <pcl/features/cgf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/distances.h>

using namespace pcl;
using namespace pcl::io;

using KdTreePtr = search::KdTree<PointXYZ>::Ptr;

// PointCloud<PointXYZ> cloud;
using PointInT = PointXYZ;
using PointOutT = CGFSignature16;
static PointCloud<PointInT>::Ptr cloud (new PointCloud<PointInT> ());
IndicesPtr indices;
KdTreePtr tree;


// TEST (PCL, CGFEstimation)
int test()
{
  std::cout << "\ntesting\n";
  // NEXT: load denser point cloud
  // Setup
  PointInT pmin, pmax;
  getMaxSegment(*cloud, pmin, pmax);
  float cloud_diam = euclideanDistance(pmin, pmax);
  float rRF = .05 * cloud_diam;
  float rmax = .17 * cloud_diam;
  float rmin = .01 * cloud_diam;
  std::cout << "cloud_diam: " << cloud_diam << "\n" ;
  std::cout << "cloud->size(): " << cloud->size() << "\n" ;

  // Create CGFEstimation object
  CGFEstimation<PointInT, PointOutT> cgf_estimation (4,2,2, rmin, rmax, rRF);
  cgf_estimation.setInputCloud (cloud);
  PointCloud<PointOutT>::Ptr feature_cloud (new PointCloud<PointOutT> ());
  std::cout << "Created object \n" ;
  
  // Test NeuralNetwork
  cgf_estimation.setCompression("/home/zoe/catkin_ws/src/pcl/test/features/NeuralNetwork_test.csv");
  // Eigen::VectorXf input;
  // input.setOnes(16);
  // std::cout << "input: " << input;
  // Eigen::VectorXf output(16);
  // cgf_estimation.getCompression().applyNN(input, output);
  // std::cout << "output: " << output << "\n"
  
  // cgf_estimation.setRadiusSearch(cgf_estimation.rmax_); 
  PCLBase<PointInT> * ptr = &cgf_estimation;

  // Test LRF
  // int idx = rand() % cloud->size();
  int idx = cloud->size()/2;
  // PointInT pt = cloud->points[idx];
  // Eigen::Vector4f pt_vec {pt.x, pt.y, pt.z, 0.};
  // demeanPointCloud(cgf_estimation.nn_cloud_, pt_vec, cgf_estimation.nn_cloud_); 

  // std::cout << "Nearest neighbor searching... \n" ;
  // int num_RF_nns = cgf_estimation.ORsearchForNeighbors(idx, cgf_estimation.rRF_, cgf_estimation.nn_indices_RF_, cgf_estimation.nn_dists_RF_);
  // int num_nns = cgf_estimation.ORsearchForNeighbors(idx, cgf_estimation.rmax_, cgf_estimation.nn_indices_, cgf_estimation.nn_dists_);
  // std::cout << "NNs: " << num_RF_nns << ' ' << num_nns << '\n';
  // copyPointCloud(*cloud, cgf_estimation.nn_indices_, cgf_estimation.nn_cloud_);
  // std::cout << "Point cloud copied \n" ;
  // cgf_estimation.localRF();
  // std::cout << "LRF computed \n" << cgf_estimation.lrf_ << '\n';

  // Generate histogram
  // cgf_estimation.sph_hist_.setZero();
  // cgf_estimation.computeSphericalHistogram(pt);
  indices.reset(new std::vector<int>());
  indices->push_back(idx);
  indices->push_back(idx+1);
  std::cout << "Created indices \n" ;
  
  ptr -> setIndices(indices);
  std::cout << "Set indices \n" ;
  // cgf_estimation.computeCGFSignatures();
  cgf_estimation.compute(*feature_cloud);
  cgf_estimation.print_arr(feature_cloud->points[0].histogram);
  std::cout << "feature_cloud->size(): " << feature_cloud->size() << "\n" ;
  
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
  IndicesPtr indices (new Indices);
  indices->resize (cloud->size ());
  for (std::size_t i = 0; i < indices->size (); ++i)
    (*indices)[i] = static_cast<int> (i);

  tree.reset (new search::KdTree<PointXYZ> (false));
  tree->setInputCloud (cloud->makeShared ());

  return test();
  // testing::InitGoogleTest (&argc, argv);
  // return (RUN_ALL_TESTS ());
}