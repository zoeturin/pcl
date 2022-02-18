
//  #ifndef PCL_FEATURES_CGF_H_
//  #define PCL_FEATURES_CGF_H_
#pragma once
#include <pcl/features/feature.h>

namespace pcl
{
  template<typename PointInT, typename PointOutT>
  class CGFEstimation : public Feature<PointInT, PointOutT>
  {
  public:
    using Ptr = shared_ptr<CGFEstimation<PointInT, PointNT, PointOutT> >;
    using ConstPtr = shared_ptr<const CGFEstimation<PointInT, PointOutT> >;
    using Feature<PointInT, PointOutT>::feature_name_;
    using Feature<PointInT, PointOutT>::getClassName;
    using Feature<PointInT, PointOutT>::indices_;
    using Feature<PointInT, PointOutT>::search_radius_;
    using Feature<PointInT, PointOutT>::search_parameter_;
    using Feature<PointInT, PointOutT>::input_;
    using Feature<PointInT, PointOutT>::surface_;

    using PointCloudIn = pcl::PointCloud<PointInT>;
    using PointCloudInPtr = typename PointCloudIn::Ptr;
    using PointCloudOut = typename Feature<PointInT, PointOutT>::PointCloudOut;

    // Constructors:
    CGF(int az_div, int el_div, int rad_div, string file) :
      //az_div_(), el_div(), rad_div_()//, cloud_diam_ () 
    {
      // Histogram stuff:
      az_div_ = az_div;
      el_div_ = el_div;
      rad_div_ = rad_div;
      radiusThresholds();
      int N = az_div * el_div * rad_div;
      sph_hist_ = Eigen::Zero(N);

      // Compression stuff:
      

      feature_name_ = "CGFEstimation";
    };

    // Computation member function declarations
    void
      computeSphericalHistogram(std::size_t index, PointCloudOut &output); // TODO: params

    void
      computeCGFSignature(); // TODO: params

      // Define getters and setters:
    inline void
      getHistogramDivisions(int az_div, int el_div, int rad_div)
    {
      az_div = az_div_;
      el_div = el_div_;
      rad_div = rad_div_;
    }

    // inline void
    //   setHistogramDivisions(int& az_div, int& el_div, int& rad_div)
    // {
    //   az_div_ = az_div;
    //   el_div_ = el_div;
    //   rad_div_ = rad_div;
    //   // TODO: new sph_hist_ of correct size
    // }

    void 
      setFeatureWeights(); // TODO

  protected:
    
    //////////////////////////////////// Local Reference Functions ////////////////////////////////////
    void
      disambiguateRF(Eigen::Matrix3f &eig);

    void
      localRF();

    //////////////////////////////////// Histogram Functions ////////////////////////////////////
    inline float
      azimuth(PointInT pt);

    inline float
      elevation(PointInT pt);

    void
      radiusThresholds();

    int
      threshold(float val, float divs);

    int
      threshold(float val, const Eigen::VectorXf thresholds_vec);

    int
      getBin(PointInT pt);

    void
      computeSphericalHistogram(const PointInT &pt, PointCloud<PointInT> &nn_cloud);

    //////////////////////////////////// Feature Computation ////////////////////////////////////

    void 
      computeSignature();

    void
      computeCGFSignature(const PointInT &pt, PointCloud<PointInT> &nn_cloud);
      
    /** \brief Estimate the set of all CGF (Compact Geometric Feature) signatures for the input cloud
      *
      */
    void
      computeCGFSignatures();

    void
      computeFeature(PointCloudOut& output) override;

    // for（int i = 0； i < cornerCloud->points.size(); i++）{
    //     // clear current feature
    //     computeCGF(cornerCloud->points[i], currentFeature);
    //     featureCloud->push_back(currentFeature);
    // }

    //////////////////////////////////// Members ////////////////////////////////////
    // Histogram parameters:
    const int az_div_, el_div_, rad_div_;
    const Eigen::VectorXf rad_thresholds_;
    float rmin_, rmax_; // TODO: set

    Eigen::VectorXi sph_hist_;
    Eigen::Matrix3f cov_;
    Eigen::Matrix3f eig_;

    std::vector<int> nn_indices_;
    std::vector<float> nn_dists_;

    Eigen::Affine3f transformation_;
    pcl::PointCloud<pcl::PointInT>::Ptr nn_cloud_;


    

    PointCloudOut output_;
  };
}

//  #endif // PCL_FEATURES_CGF_H_
