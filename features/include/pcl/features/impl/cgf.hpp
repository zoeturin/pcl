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
    int weight_idx = 0;
    for (auto idx : indices)
    {
      Eigen::Vector4f pt;
      pt[0] = cloud.points[idx].x; //- centroid[0];
      pt[1] = cloud.points[idx].y; //- centroid[1];
      pt[2] = cloud.points[idx].z; //- centroid[2];

      covariance_matrix(1, 1) += pt.y() * pt.y() * weights(weight_idx);
      covariance_matrix(1, 2) += pt.y() * pt.z() * weights(weight_idx);
      covariance_matrix(2, 2) += pt.z() * pt.z() * weights(weight_idx);

      pt *= pt.x();
      covariance_matrix(0, 0) += pt.x() * weights(weight_idx);
      covariance_matrix(0, 1) += pt.y() * weights(weight_idx);
      covariance_matrix(0, 2) += pt.z() * weights(weight_idx);
      weight_idx++;
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
    // OPT: make this more efficient? rather than alloc new VectorXf, could loop over nn_dists and modify
    // std::cout << "calculating LRF weights \n" ;
    Eigen::VectorXf weights = rRF_ - Eigen::Map<Eigen::ArrayXf>(nn_dists_RF_.data(), nn_dists_RF_.size()); // Assume nn_distsRF is same length as indices
    
    // std::cout << "computing covariance \n" ;
    computeWeightedCovarianceMatrix(nn_cloud_, nn_indices_RF_, cov_, weights);
    // Get eigenvectors of weighted covariance matrix
    Eigen::Vector3f eigen_values;
    // std::cout << "computing eigenvectors \n" ;
    pcl::eigen33(cov_, lrf_, eigen_values); // eigenvales in increasing order
    lrf_.colwise().reverseInPlace(); // want eigenvectors in order of decreasing eigenvalues
    // std::cout << "disambiguating \n" ;
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
    while (val >= thresholds_vec(bin) && bin < thresholds_vec.size()-1) bin++; // OPT More efficient way to do this? binary search?
    return bin;
  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::radiusThresholds()
  {
    rad_thresholds_ = Eigen::VectorXf::LinSpaced(rad_div_, 0, rad_div_ - 1);
    rad_thresholds_ = exp( ( rad_thresholds_ / rad_div_ * log(rmax_ / rmin_) ).array() + log(rmin_));
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
  pcl::CGFEstimation<PointInT, PointOutT>::computeSphericalHistogram(const PointInT& pt)
  {
    // Initialization
    sph_hist_.setZero();

    // Transform neighbors (q) to have relative positions q - pt
    Eigen::Vector4f pt_vec {pt.x, pt.y, pt.z, 0.};
    demeanPointCloud(nn_cloud_, pt_vec, nn_cloud_); // can pass in any pt as "centroid"

    // Local RF
    localRF();

    // Transform neighboring points into LRF
    transformation_.setIdentity();
    transformation_.linear() = lrf_;
    transformation_.inverse();
    transformPointCloud(nn_cloud_, nn_cloud_, transformation_);

    // iterate through points and increment bins
    for (std::size_t idx = 0; idx < nn_cloud_.size(); idx++)
    {
      // get bin index
      int bin = getBin(nn_cloud_[idx]);
      sph_hist_(bin) += 1;
    }
    sph_hist_ /= float(nn_cloud_.size());
  }

  //////////////////////////////////// Feature Computation ////////////////////////////////////

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeCGFSignature(const PointInT& pt)
  {
    // Compute spherical histogram
    computeSphericalHistogram(pt);
    std::cout << "sph_hist_: " << sph_hist_ << "\n" ;

    // Compress histogram using learned weights 
    std::cout << "Applying Neural Network \n" ;
    compression_.applyNN(sph_hist_, signature_);
  }

  template <typename PointInT, typename PointOutT> int // TEMP fn for externally checking nearest neighbor search
  pcl::CGFEstimation<PointInT, PointOutT>::ORsearchForNeighbors(
    int idx, 
    float radius, 
    std::vector<int>& nn_indices, 
    std::vector<float>& nn_dists)
  {
    this -> initCompute ();
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
      std::size_t pt = (*indices_)[idx]; // need actual index of point of interest in cloud
      std::cout << "First nearest neighbors search \n" ;
      std::cout << "Input cloud size: " << input_->size() << "\n" ;

      int num_nns = this->searchForNeighbors(pt, search_parameter_, nn_indices_, nn_dists_); // do NN search for histogram generation (range search <rmax_)
      std::cout << "Point of interest: " << input_->points[pt] << "\n" ;
      std::cout << "Number of neighbors for histogram: " << num_nns << "\n" ;
      copyPointCloud(*input_, nn_indices_, nn_cloud_);

      // OPT: More efficient to iterate over NNs or do second NN search?
      std::cout << "Second nearest neighbors search \n" ;
      nn_indices_RF_.clear();
      nn_dists_RF_.clear();

      for (int i = 0; i < nn_cloud_.size(); i++)
      {
        if (nn_dists_[i] < pow(rRF_,2) && nn_dists_[i] > 0)
        {
          nn_indices_RF_.push_back(i); // idx wrt nn_cloud_
          nn_dists_RF_.push_back(sqrt(nn_dists_[i])); // OPT: optimize by just indexing nn_dists_ later?
        }
      }

      if (nn_indices_RF_.size() < 5) // fewer than 5 neighbors: can't make feature, // LATER: add as parameter somewhere
      {
        std::cout << "Too few neighbors for LRF generation \n" ;
        for (Eigen::Index d = 0; d < sph_hist_.size(); ++d)
          output_.points[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN();

        output_.is_dense = false;
        continue;
      }
      
      std::cout << "Computing signature \n" ;
      computeCGFSignature(input_->points[pt]);
      std::cout << "Copying signature to output field \n" ;
      std::copy(signature_.data(), signature_.data() + signature_.size(), output_.points[idx].histogram); // histogram is preallocated
      std::cout << "signature: \n" ;
      print_arr(output_.points[idx].histogram);

      output_.points[idx].x = input_->points[idx].x;
      output_.points[idx].y = input_->points[idx].y;
      output_.points[idx].z = input_->points[idx].z;
      std::cout << "output point x,y,z: " << output_.points[idx].x << ' ' << output_.points[idx].y << ' ' << output_.points[idx].z << "\n" ;
    }

  }

  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::computeFeature(PointCloudOut& output) 
  {
    if ( !compression_.initialized() )
      throw UninitializedException(
        "Compression weights and biases have not been set. Use setCompression() member function before feature computation.");
    std::cout << "Calling computeFeature \n" ;
    output_ = output;
    // std::cout << "init compute: \n" ;
    // this -> initCompute ();
    std::cout << "Computing signatures \n" ;
    computeCGFSignatures();

  }


  //////////////////////////////////// I/O for weights/biases ////////////////////////////////////
  template <typename PointInT, typename PointOutT> void
  pcl::CGFEstimation<PointInT, PointOutT>::readMatrices(
    std::vector<MatPtr>& weights, 
    std::vector<MatPtr>& biases, 
    std::string file_str)
  {

    MatPtr matrix( new Eigen::MatrixXf (1,1));
    std::ifstream fileStream(file_str);
    std::string rowString;
    std::string elem;

    int colIdx = 0;
    int rowIdx = 0;
    auto strip_whitespace = [] (std::string &str) -> bool { str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end()); return true;};
    // LATER: make more robust, add support for tensorflow and/or julia models? currently: depends on whitespace, etc; assumes matrices have more than one row, biases don't
    
    if (!fileStream.is_open()) {
        std::cerr << "Could not open the file - '"
             << file_str << "'" << std::endl;
        exit(EXIT_FAILURE);
    }


    while ( std::getline(fileStream, rowString) ) // for each line of file:
    {
      std::stringstream rowStream(rowString); 
      colIdx = 0;

      while (getline(rowStream, elem, ',') && strip_whitespace(elem) && !elem.empty()) // read elems of line, comma separated
      {
        matrix -> conservativeResize(rowIdx+1, std::max(colIdx+1, (int) matrix->cols()));
        (*matrix)(rowIdx, colIdx) =  std::stof(elem); // coeff doesn't perform range checking but due to resize should be guaranteed in range
        colIdx++;
      }
      
      if (colIdx == 0 || fileStream.eof()) // when reached the end of a matrix (line is whitespace)
      {
        std::cout << "\n\nmatrix: \n" << *matrix << "\n\n";
        MatPtr new_ptr = move(matrix); // want to transfer ownership of matrix to new ptr 

        if (rowIdx > 1)
        {
          weights.push_back(new_ptr);
        }
        else{
          new_ptr->transposeInPlace(); // LATER: enforce column vector
          biases.push_back(new_ptr);
        }
        matrix = MatPtr(new Eigen::MatrixXf); 
        rowIdx = 0;
        matrix->setZero(1,1);
      }
      else
      {
        rowIdx++; // if line isn't whitespace, increment number of rows
      }
    }
    // LATER: add some checks? eg matrix dimensions (some checks done in Layer class), same number of bias and weight matrices, or do elsewhere idk
  }

  //////////////////////////////////// Other Tools ////////////////////////////////////
  template <typename PointInT, typename PointOutT>
  template <typename T, size_t N> void
  pcl::CGFEstimation<PointInT, PointOutT>::print_arr(T (&arr)[N])
      {
        for (int i = 0; i<N; i++)
        {
          std::cout << arr[i] << ' ';
        }
        std::cout << "\n" ;
      }      

  // template <typename Elem> void
    //   print_arr(const std::vector<Elem> arr)
    //   {
    //     std::cout <<'\n';
    //     for (Elem elem : arr)
    //     {
    //       std::cout << elem << ' ';
    //     }
    //     std::cout << "\n" ;
    //   }
}

#define PCL_INSTANTIATE_CGFEstimation(T1, T2) template class PCL_EXPORTS pcl::CGFEstimation<T1, T2>;