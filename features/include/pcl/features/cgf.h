
#pragma once
#define PCL_NO_PRECOMPILE
#include <pcl/features/feature.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
     
namespace pcl
{
  class Layer
  {
    public:
      Layer();

      Layer(Eigen::MatrixXf weights, Eigen::VectorXf biases)
      {
        setWeightsAndBiases(weights, biases);
        set_activation_relu(1.0); // LATER: make possible to set from constructor
      }

      void
      setWeightsAndBiases(Eigen::MatrixXf &weights, Eigen::VectorXf &biases) 
      {
        if (weights.rows() != biases.size())
            throw std::invalid_argument ("Weight matrix must have same number of rows as bias vector");
        weights_ = weights; 
        biases_ = biases;
        input_size_ = weights.cols();
        output_size_ = weights.rows();
      }

      void
      getWeightsAndBiases(Eigen::MatrixXf &weights, Eigen::VectorXf &biases)
      {
        weights = weights_; // ??: transfer ownership...?
        biases = biases_;
      }

      int inputSize()
      {
        return input_size_;
      }

      int outputSize()
      {
        return output_size_;
      }

      // void
      // setActivation(std::function<float (float)> activation, string name) 
      // {   
      //     activation_name_ = name;
      //     activation_ = activation;
      // }

      // LATER: get activation

      Eigen::VectorXf 
      applyLayer(const Eigen::VectorXf &input) // ??: inline?
      {
        output_ = (weights_ * input + biases_).unaryExpr(std::ref(activation_));
        return output_; // ?? is this bad practice?
      }

      // LATER: add typical activation functions

      void
      set_activation_relu(float slope)
      {
        activation_name_ = "relu";
        activation_ = [slope] (float x) -> float {return x>0 ? x*slope : 0;}; 
      }

      // void
      // getOutput(Eigen::VectorXf &output)
      // {
      //   output_ = output;
      // }

      private:
        int input_size_;
        int output_size_;
        Eigen::VectorXf output_;
        Eigen::MatrixXf weights_;
        Eigen::VectorXf biases_;
        std::function<float (float)> activation_; 
        std::string activation_name_;
  };

  class NeuralNetwork
  {
    public:
    // ??: want vector of shared pointers instead? not sure what Eigen does internally in terms of copying/assignment
    NeuralNetwork(const std::vector<Eigen::MatrixXf>& weights, const std::vector<Eigen::MatrixXf>& biases) 
    {
      if (weights.size() != biases.size()) throw std::invalid_argument ("Must have same number of weight matrices as bias vectors");
      num_layers_ = weights.size();
      for (int i = 0; i < num_layers_; i++)
      {
        layers_.push_back(Layer(weights[i], biases[i]));
        layer_sizes_.push_back(layers_.back().outputSize()); 
        if (i>0 && layers_[i-1].outputSize() != layers_[i].inputSize()) // LATER: move this code block elsewhere, eg through custom error type?
        {
          std::ostringstream err_stream;
          err_stream << "Output size of layer " << i-1 << " (" << layers_[i-1].outputSize() << " )";
          err_stream << " does not match input size of layer " << i << " (" << layers_[i].inputSize() << " )";
          throw std::invalid_argument (err_stream.str());
        }
      }
      input_size_ = layers_.front().inputSize();
      output_size_ = layers_.back().outputSize();
    }

    void 
      applyNN(const Eigen::VectorXf& input, Eigen::VectorXf& output)
      {
        const Eigen::VectorXf *temp = &input; // ?? having this as const might cause issues with assigning output/casting to non-const?
        for (int i = 0; i < num_layers_; i++)
        {
          temp = & layers_[i].applyLayer(*temp); // ?? idk what I'm doing
        }
        output = *temp;
      }

    int 
      getOutputSize() 
      {
        return output_size_;
      }

    private:
      int num_layers_;
      int input_size_;
      int output_size_;
      std::vector<int> layer_sizes_; // vector of output sizes of each layer
      std::vector<Layer> layers_;
  };

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

    void 
      MatrixXd readMatrices(string file_str, int num_layers);

    //////////////////////////////////// Constructors ////////////////////////////////////

    CGF(int az_div, int el_div, int rad_div, std::string file_str) :
    {
      // Histogram stuff:
      az_div_ = az_div;
      el_div_ = el_div;
      rad_div_ = rad_div;
      radiusThresholds();
      int N = az_div * el_div * rad_div;
      sph_hist_ = Eigen::Zero(N);

      // Compression stuff:
      setCompression(std::string file_str);

      feature_name_ = "CGFEstimation";
    };  

    void
      readMatrices(std::vector<Eigen::MatrixXf>& weights, std::vector<Eigen::MatrixXf>& biases, std::string file_str);

    //////////////////////////////////// Getters and Setters ////////////////////////////////////

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
      setCompression(std::string file_str)
      {
        std::vector<Eigen::MatrixXf> weights;
        std::vector<Eigen::MatrixXf> biases;
        readMatrices(&weights, biases, file_str);
        compression_ = NeuralNetwork(weights, biases);
        if compression_.getOutputSize() != PointOutT.descriptorSize() 
        {
          std::ostringstream err_stream;
          err_stream << "Output size of neural network ( " << compression_.getOutputSize() << " )";
          err_stream << " does not match dimensionality of feature ( " << PointOutT.descriptorSize() << " )";
          throw std::invalid_argument (err_stream.str());
        }
      }

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
      computeCGFSignature(const PointInT &pt, PointCloud<PointInT> &nn_cloud);
      
    /** \brief Estimate the set of all CGF (Compact Geometric Feature) signatures for the input cloud
      *
      */
    void
      computeCGFSignatures();

    void
      computeFeature(PointCloudOut& output) override;


    //////////////////////////////////// Member Variables ////////////////////////////////////

    // Histogram parameters:
    const int az_div_, el_div_, rad_div_;
    const Eigen::VectorXf rad_thresholds_;
    float rmin_, rmax_; // TODO: set

    Eigen::VectorXf sph_hist_;
    Eigen::Matrix3f cov_;
    Eigen::Matrix3f eig_;

    std::vector<int> nn_indices_;
    std::vector<float> nn_dists_;

    Eigen::Affine3f transformation_;
    pcl::PointCloud<pcl::PointInT>::Ptr nn_cloud_;

    pcl::NeuralNetwork compression_;

    PointCloudOut output_;
  };
}

