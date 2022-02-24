#pragma once
#include <pcl/features/cgf.h>


namespace pcl {
  //////////////////////////////////// LRF Tools ////////////////////////////////////
  // Modify computeCovarianceMatrix to accept weights
  // Could use class VectorAverage instead, currently only supports centroid automatically calculated as mean of points
  template <typename PointT, typename Scalar> inline unsigned int
    computeWeightedCovarianceMatrix(const pcl::PointCloud<PointT>& cloud,
      const Eigen::Matrix<Scalar, 4, 1>& centroid,
      Eigen::Matrix<Scalar, 3, 3>& covariance_matrix,
      const Eigen::VectorXf& weights)
  {
    int size = cloud.size();
    if (cloud.empty ()) // shouldn't get here if empty
     return (0);

    // Initialize to 0
    covariance_matrix.setZero();

    std::size_t point_count;
    
    // If the data is dense, we don't need to check for NaN
    // if (cloud.is_dense)
    // {
      point_count = static_cast<unsigned> (cloud.size ());
      // For each point in the cloud
      for (int idx = 0; idx < size; idx++)
      {
        Eigen::Matrix<Scalar, 4, 1> pt;
        pt[0] = cloud.points[idx].x - centroid[0];
        pt[1] = cloud.points[idx].y - centroid[1];
        pt[2] = cloud.points[idx].z - centroid[2];

        covariance_matrix(1, 1) += pt.y() * pt.y() * weights(idx);
        covariance_matrix(1, 2) += pt.y() * pt.z() * weights(idx);
        covariance_matrix(2, 2) += pt.z() * pt.z() * weights(idx);

        pt *= pt.x();
        covariance_matrix(0, 0) += pt.x() * weights(idx);
        covariance_matrix(0, 1) += pt.y() * weights(idx);
        covariance_matrix(0, 2) += pt.z() * weights(idx);
      }
    // }
    // NaN or Inf values could exist => check for them
    // else // shouldn't be the case for CGF usage // ?? does it really save anything *not* to check for NaN? ie why have 2 cases for dense/not dense
    // { // LATER: add back if/else block once if block works
    //   point_count = 0;
    //   // For each point in the cloud
    //   for (int idx = 0; idx < size; idx++)
    //   {
    //     // Check if the point is invalid
    //     if (!isFinite(cloud[index]))
    //       continue;

    //     Eigen::Matrix<Scalar, 4, 1> pt;
    //     pt[0] = cloud.points[index].x - centroid[0];
    //     pt[1] = cloud.points[index].y - centroid[1];
    //     pt[2] = cloud.points[index].z - centroid[2];

    //     covariance_matrix(1, 1) += pt.y() * pt.y() * weights(idx);
    //     covariance_matrix(1, 2) += pt.y() * pt.z() * weights(idx);
    //     covariance_matrix(2, 2) += pt.z() * pt.z() * weights(idx);

    //     pt *= pt.x();
    //     covariance_matrix(0, 0) += pt.x() * weights(idx);
    //     covariance_matrix(0, 1) += pt.y() * weights(idx);
    //     covariance_matrix(0, 2) += pt.z() * weights(idx);
    //     ++point_count;
    //   }
    // }
    covariance_matrix(1, 0) = covariance_matrix(0, 1);
    covariance_matrix(2, 0) = covariance_matrix(0, 2);
    covariance_matrix(2, 1) = covariance_matrix(1, 2);
    covariance_matrix /= point_count; // normalize
    return (static_cast<unsigned int> (point_count));
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::disambiguateRF(Eigen::Matrix3f& eig)
  {
    //TODO
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::localRF(PointCloud<PointInT>& nn_cloud)
  {
    // Get weighted covariance of neighboring points
    // float* ptr_data = &nn_dists_[0];
    // Eigen::VectorXf v2 = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(v1.data(), v1.size());
    Eigen::VectorXf weights = search_radius_ - Eigen::Map<Eigen::ArrayXf>(nn_dists_.data(), sizeof(nn_dists_)); // FIX
    computeWeightedCovarianceMatrix<PointInT, float>(nn_cloud, Eigen::Vector4f::Zero(), cov_, weights);
    // Get eigenvectors of weighted covariance matrix
    Eigen::Vector3f eigen_value;
    pcl::eigen33(cov_, eig_, eigen_value); // eigenvales in increasing order
    eig_.colwise().reverseInPlace(); // want eigenvectors in order of decreasing eigenvalues
    disambiguateRF(eig_);
  }

  //////////////////////////////////// Histogram Tools ////////////////////////////////////

  template <typename PointInT, typename PointOutT> inline float
  pcl::CGFEstimation<PointInT, PointOutT>::azimuth(PointInT pt)
  {
    return atan2(pt.y, pt.x) + M_PI;
  }

  template <typename PointInT, typename PointOutT> inline float
  pcl::CGFEstimation<PointInT, PointOutT>::elevation(PointInT pt)
  {
    return atan2(pt.z, sqrt(pow(pt.x, 2) + pow(pt.y, 2))) + M_PI;
  }

  template <typename PointInT, typename PointOutT> inline float
  pcl::CGFEstimation<PointInT, PointOutT>::radius(PointInT pt)
  {
    return sqrt(pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2));
  }

  template <typename PointInT, typename PointOutT> int
  pcl::CGFEstimation<PointInT, PointOutT>::threshold(float val, float divs)
  {
    int bin = (int)(val / divs * 2 * M_PI);
    return (bin >= 2 * M_PI ? bin - 1 : bin); // Account for case where val == 2*PI
  }

  template <typename PointInT, typename PointOutT> int
  pcl::CGFEstimation<PointInT, PointOutT>::threshold(float val, const Eigen::VectorXf thresholds_vec)
  {
    int bin = 0;
    while (val >= thresholds_vec(bin)) bin++; // ?? More efficient way to do this?
    return bin;
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::radiusThresholds()
  {
    rad_thresholds_ = Eigen::VectorXf::LinSpaced(rad_div_, 0, rad_div_ - 1);
    rad_thresholds_ = exp((rad_thresholds_.array() / rad_div_) * log(rmax_ / rmin_) + log(rmin_));
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
  pcl::CGFEstimation<PointInT, PointOutT>::computeSphericalHistogram(const PointInT& pt, PointCloud<PointInT>& nn_cloud)
  {
    // ??: better to pass in outputs by reference and modify, or just modify class member variables?
    // Initialization
    sph_hist_.setZero();

    // Transform neighbors (q) to have relative positions q - pt
    // transformation_.setIdentity();
    // transformation_.translation() << -pt.x, -pt.y, -pt.z;

    // transformPointCloud(nn_cloud, nn_cloud, transformation_);
    Eigen::Vector4f pt_vec {pt.x, pt.y, pt.z, 0.};
    demeanPointCloud(nn_cloud, pt_vec, nn_cloud); // can pass in any pt as "centroid"; ?? ok to have input same as output? (eg allocating?)

    // Local RF
    localRF(nn_cloud);

    // Transform neighboring points into LRF
    transformation_.setIdentity();
    transformation_.linear() = eig_;
    transformation_.inverse();
    transformPointCloud(nn_cloud, nn_cloud, transformation_);

    // iterate through points and increment bins
    for (std::size_t idx = 0; idx < nn_cloud.size(); idx++)
    {
      // get bin index
      idx = getBin(nn_cloud[idx]);
      sph_hist_(idx)++;
    }
    sph_hist_ /= nn_cloud.size();
  }

  //////////////////////////////////// Feature Computation ////////////////////////////////////

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignature(const PointInT& pt, PointCloud<PointInT>& nn_cloud)
  {
    // Compute spherical histogram
    computeSphericalHistogram(pt, nn_cloud);

    // Compress histogram using learned weights 
    compression_.applyNN(sph_hist_, signature_);
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignatures()
  {
    // iterate through input cloud
    // Iterate over the entire index vector
    for (std::size_t idx = 0; idx < indices_->size(); ++idx)
    {
      // NN range search 
      // ?? if nn_indices_ and nn_dists_ properly reset by calls to searchForNeighbors
      if (this->searchForNeighbors(idx, search_parameter_, nn_indices_, nn_dists_) < 5) // fewer than 5 neighbors: can't make feature, // ??: increase?
      {
        for (Eigen::Index d = 0; d < sph_hist_.size(); ++d)
          output_[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN();

        output_.is_dense = false;
        continue;
      }
      // nn_cloud_.reset(); // ?? necessary before copyPointCloud? don't think so?  
      copyPointCloud(*input_, nn_indices_, nn_cloud_);
      computeCGFSignature(input_->points[idx], nn_cloud_);
      std::copy(signature_.data(), signature_.data() + signature_.size(), output_.points[idx].histogram); // histogram is preallocated
      output_.points[idx].x = input_->points[idx].x;
      output_.points[idx].y = input_->points[idx].y;
      output_.points[idx].z = input_->points[idx].z;
    }

  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeFeature(PointCloudOut& output) // ?? need unused arg to override?
  {
    computeCGFSignatures();

  }


  //////////////////////////////////// I/O for weights/biases ////////////////////////////////////
  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::readMatrices(std::vector<Eigen::MatrixXf>& weights, std::vector<Eigen::MatrixXf>& biases, std::string file_str)
  {
    // ??: should weights and biases vectors be passed via pointer instead?
    // ??: should these be vectors of pointers instead?
    Eigen::MatrixXf matrix; // temporary matrix storage

    std::ifstream fileStream(file_str);
    std::string rowString;
    std::string elem;

    int colIdx = 0;
    int rowIdx = 0;

    // LATER: not very robust, depends on whitespace, etc; assumes matrices have more than one row, biases don't
    while (getline(fileStream, rowString)) // for each line of file:
    {

      std::stringstream rowStream(rowString); 
      colIdx = 0;
      while (getline(rowStream, elem, ',')) // read elems of line, comma separated
      {
        matrix(rowIdx, colIdx) = std::stof(elem);
        colIdx++;
      }
      
      if (colIdx == 0) // when reached the end of a matrix (line is whitespace)
      {
        if (rowIdx > 1)
        {
          weights.push_back(matrix); // ??: moves matrix ownership in this case?
        }
        else{
          biases.push_back(matrix);
        }
        matrix = Eigen::MatrixXf::Zero(0,0); // ?? want to reset to size (0,0) dynamically allocated matrix
        rowIdx = 0;
      }
      else
      {
        rowIdx++; // if line isn't whitespace, increment number of rows
      }
    }
    // LATER: add some checks? eg matrix dimensions (some checks done in Layer class), same number of bias and weight matrices, or do elsewhere idk
  }

}

#define PCL_INSTANTIATE_CGFEstimation(T1, T2) template class PCL_EXPORTS pcl::CGFEstimation<T1, T2>;