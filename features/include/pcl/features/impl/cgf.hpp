#pragma once
#include <pcl/features/cgf.h>
#include <Eigen>

//////////////////////////////////// LRF Tools ////////////////////////////////////
// Modify computeCovarianceMatrix to accept weights
// Could use class VectorAverage instead, currently only supports centroid automatically calculated as mean of points
namespace pcl{
 template <typename PointT, typename Scalar> inline unsigned int
 computeWeightedCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                          const Indices &indices,
                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                          const Eigen::VectorXf &weights)
 {
   if (indices.empty ())
     return (0);
  
   // Initialize to 0
   covariance_matrix.setZero ();
  
   std::size_t point_count;
   // If the data is dense, we don't need to check for NaN
   if (cloud.is_dense)
   {
     point_count = 0;
     // For each point in the cloud
     for (const auto& idx: indices)
     {
       Eigen::Matrix<Scalar, 4, 1> pt;
       pt[0] = cloud[idx].x - centroid[0];
       pt[1] = cloud[idx].y - centroid[1];
       pt[2] = cloud[idx].z - centroid[2];
  
       covariance_matrix (1, 1) += pt.y () * pt.y () * weights(point_count);
       covariance_matrix (1, 2) += pt.y () * pt.z () * weights(point_count);
  
       covariance_matrix (2, 2) += pt.z () * pt.z () * weights(point_count);
  
       pt *= pt.x ();
       covariance_matrix (0, 0) += pt.x () * weights(point_count);
       covariance_matrix (0, 1) += pt.y () * weights(point_count);
       covariance_matrix (0, 2) += pt.z () * weights(point_count);
       ++point_count;
     }
   }
   // NaN or Inf values could exist => check for them
   else
   {
     point_count = 0;
     // For each point in the cloud
     for (const auto &index : indices)
     {
       // Check if the point is invalid
       if (!isFinite (cloud[index]))
         continue;
  
       Eigen::Matrix<Scalar, 4, 1> pt;
       pt[0] = cloud[index].x - centroid[0];
       pt[1] = cloud[index].y - centroid[1];
       pt[2] = cloud[index].z - centroid[2];
  
       covariance_matrix (1, 1) += pt.y () * pt.y () * weights(point_count);
       covariance_matrix (1, 2) += pt.y () * pt.z () * weights(point_count);
  
       covariance_matrix (2, 2) += pt.z () * pt.z () * weights(point_count);
  
       pt *= pt.x ();
       covariance_matrix (0, 0) += pt.x () * weights(point_count);
       covariance_matrix (0, 1) += pt.y () * weights(point_count);
       covariance_matrix (0, 2) += pt.z () * weights(point_count);
       ++point_count;
     }
   }
   covariance_matrix (1, 0) = covariance_matrix (0, 1);
   covariance_matrix (2, 0) = covariance_matrix (0, 2);
   covariance_matrix (2, 1) = covariance_matrix (1, 2);
   return (static_cast<unsigned int> (point_count));
 }
}

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::disambiguateRF(Eigen::Matrix3f &eig)
{
  //TODO
}

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::localRF()
{
  // Get weighted covariance of neighboring points
  Eigen::VectorXf weights = search_radius_ - nn_dists_; 
  computeWeightedCovarianceMatrix(cloud, nn_indices_, Eigen::Matrix<Scalar, 4, 1> Eigen::Zero(), cov_, weights);
  // Get eigenvectors of weighted covariance matrix
  EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
  // EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
  pcl::eigen33 (cov_, eigen_value, eig_); // eigenvales in increasing order
  eig_ = eig_.colwise().reverseInPlace(); // want eigenvectors in order of decreasing eigenvalues
  disambiguateRF(eig_);
}

//////////////////////////////////// Histogram Tools ////////////////////////////////////

template <typename PointInT, typename PointOutT> float
pcl::CGFEstimation<PointInT, PointOutT>::azimuth(PointInT pt)
{
  return atan2(pt.y, pt.x) + M_PI;
}

template <typename PointInT, typename PointOutT> float
pcl::CGFEstimation<PointInT, PointOutT>::elevation(PointInT pt)
{
  return atan2(pt.z, sqrt(pt.x**2 + pt.y**2)) + M_PI;
}

template <typename PointInT, typename PointOutT> int
pcl::CGFEstimation<PointInT, PointOutT>::threshold(float val, float divs)
{
  int bin = (int) (val/divs*2*M_PI);
  return (bin >= 2*M_PI ? bin-1 : bin); // Account for case where val == 2*PI
}

template <typename PointInT, typename PointOutT> int
pcl::CGFEstimation<PointInT, PointOutT>::threshold(float val, const Eigen::VectorXf thresholds_vec)
{
  int bin = 0;
  while (bin >= thresholdsvec(bin)) bin++;
  return bin;
}

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::radiusThresholds()
{
  rad_thresholds_ = Eigen::LinSpaced(rad_div_, 0, rad_div_-1);
  rad_thresholds_ = exp((rad_thresholds.array() / rad_div_) * log(rmax_/rmin_) + log(rmin_));
}

template <typename PointInT, typename PointOutT> int
pcl::CGFEstimation<PointInT, PointOutT>::getBin(PointInT pt)
{
  // azimuthal bin
  float az = azimuth(pt);
  int az_bin = threshold(az, az_div_);
  // elevation bin
  float el = elevation(pt);
  int el_bin = threshold(el, el_div_);
  // radial bin
  float rad = radius(pt);
  int rad_bin = threshold(rad, rad_thresholds_);
  // vectorized bin // rad, az, el
  int vec_bin = rad_bin + az_bin * rad_div_ + el_bin * (rad_div_ * az_div_);
  return vec_bin;
}

// CGF member function definitions for feature computation:
template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::computeSphericalHistogram()
{
  // Initialization
  int idx;
  sph_hist_.setZero();

  // Transform neighbors (q) to have relative positions q - pt
  transformation_.setIdentity();
  transformation_.translation() << -pt.x, -pt.y, -pt.z;
  // nn_cloud_.reset(); // UNSURE if necessary before copyPointCloud, don't think so?  
  copyPointCloud(cloud, nn_indices_, nn_cloud_);
  transformPointCloud (*cloud, *nn_cloud_, transformation_);

  // Local RF
  localRF();

  // Transform neighboring points into LRF
  transformation_.setIdentity();
  transformation_.matrix() = eig_;
  transformation_.inverse();
  transformPointCloud (*cloud, *nn_cloud_, transformation_);
  
  // iterate through points and increment bins
  for (std::size_t idx = 0; idx < nn_indices_ -> size(); idx++) 
  {
    // get bin index
    idx = getBin();
    sph_hist_(idx) ++;
  }

}

//////////////////////////////////// Feature Computation ////////////////////////////////////

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignature(PointInT pt)
{
  // TODO

  // Compute spherical histogram
  computeSphericalHistogram();

  // Compress histogram using learned weights 

}

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignatures()
{
  // iterate through input cloud
  // Iterate over the entire index vector
    for (std::size_t idx = 0; idx < indices_->size (); ++idx) 
    {
      // NN range search 
      // UNSURE if nn_indices_ and nn_dists_ properly reset by calls to searchForNeighbors
      if (this->searchForNeighbors (index, search_parameter_, nn_indices_, nn_dists_) < 5) // fewer than 5 neighbors: can't make feature, // UNSURE: increase?
        {
          for (Eigen::Index d = 0; d < sph_hist_.size (); ++d)
            output_[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN ();

          output_.is_dense = false;
          continue;
        }
      nn_cloud_ = 
      computeCGFSignature();
    }
  
}

template <typename PointInT, typename PointOutT> void
pcl::CGFEstimation<PointInT, PointOutT>::computeFeature(PointCloudOut& output)
{
  computeCGFSignatures();

}



