#pragma once
#include <pcl/features/cgf.h>


namespace pcl {
  //////////////////////////////////// LRF Tools ////////////////////////////////////
  // Modified computeCovarianceMatrix to accept weights
  // Could use class VectorAverage instead, currently only supports centroid automatically calculated as mean of points
  template <typename PointInT, typename PointOutT> unsigned int
  pcl::CGFEstimation<PointInT, PointOutT>::computeWeightedCovarianceMatrix(
      const pcl::PointCloud<PointInT>& cloud,
      const Indices& indices,
      Eigen::Matrix3f& covariance_matrix,
      const Eigen::VectorXf& weights)
  {
    // Initialize to 0
    covariance_matrix.setZero();

    std::size_t point_count;
    
    point_count = static_cast<unsigned> (indices.size ());
    // For each point in the cloud
    for (auto idx : indices)
    {
      Eigen::Vector4f pt;
      pt[0] = cloud.points[idx].x; //- centroid[0];
      pt[1] = cloud.points[idx].y; //- centroid[1];
      pt[2] = cloud.points[idx].z; //- centroid[2];

      covariance_matrix(1, 1) += pt.y() * pt.y() * weights(idx);
      covariance_matrix(1, 2) += pt.y() * pt.z() * weights(idx);
      covariance_matrix(2, 2) += pt.z() * pt.z() * weights(idx);

      pt *= pt.x();
      covariance_matrix(0, 0) += pt.x() * weights(idx);
      covariance_matrix(0, 1) += pt.y() * weights(idx);
      covariance_matrix(0, 2) += pt.z() * weights(idx);
    }

    covariance_matrix(1, 0) = covariance_matrix(0, 1);
    covariance_matrix(2, 0) = covariance_matrix(0, 2);
    covariance_matrix(2, 1) = covariance_matrix(1, 2);
    covariance_matrix /= point_count; // normalize
    return point_count;
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::disambiguateRF()
  {
    //TODO
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::localRF()
  {
    // Should only mutate cov_, lrf_
    // Get weighted covariance of neighboring points
    // LATER: make this more efficient? rather than alloc new VectorXf, could loop over nn_dists and modify
    Eigen::VectorXf weights = rRF_ - Eigen::Map<Eigen::ArrayXf>(nn_dists_RF_.data(), sizeof(nn_dists_RF_)); // Assume nn_distsRF is same length as indices
    computeWeightedCovarianceMatrix(nn_cloud_, nn_indices_RF_, cov_, weights);
    // Get eigenvectors of weighted covariance matrix
    Eigen::Vector3f eigen_values;
    pcl::eigen33(cov_, lrf_, eigen_values); // eigenvales in increasing order
    lrf_.colwise().reverseInPlace(); // want eigenvectors in order of decreasing eigenvalues
    disambiguateRF();
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
    int bin = (int)(val / (2 * M_PI/divs));
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
    std::cout << "rmax_: " << rmax_ << "\n" ;
    std::cout << "rmin_: " << rmin_ << "\n" ;
    rad_thresholds_ = Eigen::VectorXf::LinSpaced(rad_div_, 0, rad_div_ - 1);
    // std::cout << "1: " << log(rmax_/rmin_) << "\n" ;
    // std::cout << "2: " << ( rad_thresholds_ / rad_div_ * log(rmax_ / rmin_) ).array() + log(rmin_) << "\n" ;
    rad_thresholds_ = exp( ( rad_thresholds_ / rad_div_ * log(rmax_ / rmin_) ).array() + log(rmin_));
    std::cout << "rad_thresholds_: " << rad_thresholds_ << "\n" ;
  }

  template <typename PointInT, typename PointOutT> int
  pcl::CGFEstimation<PointInT, PointOutT>::getBin(PointInT pt)
  {
    // azimuthal bin
    float az = azimuth(pt);
    int az_bin = threshold(az, az_div_);
    // std::cout << "az: " << az << " az_bin: " << az_bin << "\n" ;
    // elevation bin
    float el = elevation(pt);
    int el_bin = threshold(el, el_div_);
    // std::cout << "el: " << el << " el_bin: " << el_bin << "\n" ;
    // radial bin
    float rad = radius(pt);
    int rad_bin = threshold(rad, rad_thresholds_);
    // NEXT: fix radial binning
    
    std::cout  << "rad: " << rad  << " rad_bin: " << rad_bin << "\n" ;
    // vectorized bin // rad, az, el
    int vec_bin = rad_bin + az_bin * rad_div_ + el_bin * (rad_div_ * az_div_);
    return vec_bin;
  }

  // CGF member function definitions for feature computation:
  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeSphericalHistogram(const PointInT& pt)
  {
    // ??: better to pass in outputs by reference and modify, or just modify class member variables?
    // Initialization
    std::cout << "Zeroing  \n" ;
    sph_hist_.setZero();

    // Transform neighbors (q) to have relative positions q - pt
    std::cout << "Demeaning point cloud \n" ;
    Eigen::Vector4f pt_vec {pt.x, pt.y, pt.z, 0.};
    demeanPointCloud(nn_cloud_, pt_vec, nn_cloud_); // can pass in any pt as "centroid"; ?? ok to have input same as output? (eg allocating?)

    // Local RF
    std::cout << "Computing LRF \n" ;
    localRF();

    // Transform neighboring points into LRF
    std::cout << "Setting and applying transform \n" ;
    transformation_.setIdentity();
    transformation_.linear() = lrf_;
    transformation_.inverse();
    transformPointCloud(nn_cloud_, nn_cloud_, transformation_);

    // iterate through points and increment bins
    std::cout << "Incrementing bins \n" ;
    std::cout << "nn cloud size: " << nn_cloud_.size() << "\n" ;
    for (std::size_t idx = 0; idx < nn_cloud_.size(); idx++)
    {
      // get bin index
      int bin = getBin(nn_cloud_[idx]);
      // std::cout << "bin: " << bin << "\n" ;
      sph_hist_(bin) += 1;
    }
    std::cout << "sph_hist_: " << sph_hist_ << "\n" ;
    sph_hist_ /= float(nn_cloud_.size());
    std::cout << "sph_hist_: " << sph_hist_ << "\n" ;

  }

  //////////////////////////////////// Feature Computation ////////////////////////////////////

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignature(const PointInT& pt)
  {
    // Compute spherical histogram
    computeSphericalHistogram(pt);
    std::cout << "sph_hist_: " << sph_hist_ << "\n" ;

    // Compress histogram using learned weights 
    compression_.applyNN(sph_hist_, signature_);
  }

  template <typename PointInT, typename PointOutT> int // TEMP fn for externally checking nearest neighbor search
  pcl::CGFEstimation<PointInT, PointOutT>::ORsearchForNeighbors(int idx, float radius, std::vector<int>& nn_indices, std::vector<float>& nn_dists)
  {
    std::cout << "init compute: \n" ;
    this -> initCompute ();
    std::cout << "search for neighbors fn \n" ;
    std::cout << "idx: " << idx << '\n';
    std::cout << "radius: " << radius << "\n" ;
    // std::cout << "nn_indices: " << nn_indices << "\n" ;
    // std::cout << "nn_dists: " << nn_dists << "\n" ;
    return this->searchForNeighbors(idx, radius, nn_indices, nn_dists);
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
      // ?? can you prevent nn_dists_ output/allocation? don't think it's needed
      std::size_t pt = (*indices_)[idx]; // CHECK: if searchForNeighbors uses indices[idx] or idx
      std::cout << "First nearest neighbors search \n" ;
      this->searchForNeighbors(pt, rmax_, nn_indices_, nn_dists_); // do NN search for histogram generation (range search <rmax_)

      // ?? I think it's more efficient to run second NN search like this rather than finding all dists < rRF_ and using these indices? maybe only true if below true
      // LATER: try to make this more efficient by only searching over smaller nn_cloud_ ? possibly would need to add idx to nn_cloud_ which might screw with some downstream stuff (eg RF generation) unless searchForNeighbors overloaded for point (rather than idx) input
      std::cout << "Second nearest neighbors search \n" ;
      if (this->searchForNeighbors(pt, rRF_, nn_indices_RF_, nn_dists_RF_) < 5) // fewer than 5 neighbors: can't make feature, // ??: increase?
      {
        std::cout << "Too few neighbors for LRF generation \n" ;
        for (Eigen::Index d = 0; d < sph_hist_.size(); ++d)
          output_.points[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN();

        output_.is_dense = false;
        continue;
      }
      // nn_cloud_.reset(); // ?? necessary before copyPointCloud? don't think so?  
      std::cout << "Copying point cloud \n" ;
      copyPointCloud(*input_, nn_indices_, nn_cloud_);
      std::cout << "Computing signature \n" ;
      computeCGFSignature(input_->points[pt]);
      std::cout << "Copying histogram to output field \n" ;
      std::copy(signature_.data(), signature_.data() + signature_.size(), output_.points[idx].histogram); // histogram is preallocated
      output_.points[idx].x = input_->points[idx].x;
      output_.points[idx].y = input_->points[idx].y;
      output_.points[idx].z = input_->points[idx].z;
    }

  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeFeature(PointCloudOut& output) // ?? need unused arg to override?
  {
    std::cout << "Calling computeFeature \n" ;
    output_ = output;
    // std::cout << "init compute: \n" ;
    // this -> initCompute ();
    std::cout << "Computing signatures \n" ;
    computeCGFSignatures();

  }


  //////////////////////////////////// I/O for weights/biases ////////////////////////////////////
  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::readMatrices(std::vector<MatPtr>& weights, std::vector<MatPtr>& biases, std::string file_str)
  // pcl::CGFEstimation<PointInT, PointOutT>::readMatrices(std::vector<Eigen::MatrixXf>& weights, std::vector<Eigen::MatrixXf>& biases, std::string file_str)
  {

    MatPtr matrix( new Eigen::MatrixXf (1,1));
    // Eigen::MatrixXf matrix; // temporary matrix storage

    std::ifstream fileStream(file_str);
    std::string rowString;
    std::string elem;

    int colIdx = 0;
    int rowIdx = 0;
    // std::cout << "reading matrices\n";
    auto strip_whitespace = [] (std::string &str) -> bool { str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end()); return true;};
    // LATER: make more robust, add support for tensorflow and/or julia models? currently: depends on whitespace, etc; assumes matrices have more than one row, biases don't
    
    if (!fileStream.is_open()) {
        std::cerr << "Could not open the file - '"
             << file_str << "'" << std::endl;
        exit(EXIT_FAILURE);
    }

    // std::cout << fileStream.eof();

    while ( std::getline(fileStream, rowString) ) // for each line of file:
    {
      std::stringstream rowStream(rowString); 
      colIdx = 0;

      while (getline(rowStream, elem, ',') && strip_whitespace(elem) && !elem.empty()) // read elems of line, comma separated
      {
        // TODO: add try/catch

        // matrix.conservativeResize(rowIdx+1, std::max(colIdx+1, (int) matrix.cols()));
        matrix -> conservativeResize(rowIdx+1, std::max(colIdx+1, (int) matrix->cols()));
        // matrix (rowIdx, colIdx) =  std::stof(elem);
        (*matrix)(rowIdx, colIdx) =  std::stof(elem); // coeff doesn't perform range checking but due to resize should be guaranteed in range

        colIdx++;
      }
      
      if (colIdx == 0 || fileStream.eof()) // when reached the end of a matrix (line is whitespace)
      {
        // std::cout << "\n\nmatrix: \n" << matrix << "\n\n";
        std::cout << "\n\nmatrix: \n" << *matrix << "\n\n";
        MatPtr new_ptr = move(matrix); // want to transfer ownership of matrix to new ptr 

        if (rowIdx > 1)
        {
          weights.push_back(new_ptr); // ??: moves matrix ownership in this case?
          // weights.push_back(matrix); // ??: moves matrix ownership in this case?
        }
        else{
          new_ptr->transposeInPlace(); // LATER: enforce column vector
          biases.push_back(new_ptr);
          // biases.push_back(matrix.transpose()); 
        }
        matrix = MatPtr(new Eigen::MatrixXf); // ?? want to reset to size (0,0) dynamically allocated matrix
        rowIdx = 0;
        matrix->setZero(1,1);
        // std::cout << matrix*;
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