
#pragma once
#define PCL_NO_PRECOMPILE
#include <pcl/features/feature.h>
#include <pcl/common/impl/io.hpp> // need for copyPointCloud()
#include <pcl/common/impl/transforms.hpp> // need for transformPointCloud()
#include <pcl/common/centroid.h> // demeanPointCloud()
#include <pcl/common/eigen.h> // eigen33()
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>

namespace pcl
{
  class Layer
  {
  public:
    Layer() {} // ?? need default constructor that can be called by CGFEstimation default constructor? (via NeuralNetwork default constructor)

    Layer(const shared_ptr<Eigen::MatrixXf>& weights, const shared_ptr<Eigen::MatrixXf>& biases)
    {
      std::cout<< "layer constructor\n";
      setWeightsAndBiases(weights, biases);
      set_activation_relu(1.0); // LATER: make possible to set from constructor
    }

    void
      setWeightsAndBiases(const shared_ptr<Eigen::MatrixXf>& weights, const shared_ptr<Eigen::MatrixXf>& biases)
    {

      if (weights->rows() != biases->size())
        throw std::invalid_argument("Weight matrix must have same number of rows as bias vector");
      std::cout << "set weights \n";
      // std::cout << weights;
      weights_ = move(weights); 
      biases_ = move(biases);
      input_size_ = weights->cols();
      output_size_ = weights->rows();
      std::cout << "finished setting layer\n";
      }

    void
      getWeightsAndBiases(Eigen::MatrixXf weights, Eigen::MatrixXf biases)
    {
      weights = *weights_; // ??: transfer ownership...?
      biases = *biases_;
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

    void
      applyLayer(const Eigen::VectorXf& input) // ??: inline?
    {
      std::cout << "trying to apply layer \n" << input << "input\n";
      // NEXT: bias vector should be transposed
      output_ = (*weights_ * input + *biases_).unaryExpr(std::ref(activation_));
      std::cout << "applied layer " << *biases_  << '\n';
    }

    // LATER: add other typical activation functions

    void
      set_activation_relu(float slope)
    {
      activation_name_ = "relu";
      activation_ = [slope](float x) -> float {return x > 0 ? x * slope : 0;};
    }

    // void
    // getOutput(Eigen::VectorXf &output)
    // {
    //   output_ = output;
    // }
    Eigen::VectorXf*
      getOutputPtr()
    {
      return &output_; // ?? is this bad practice?
    }

  private:
    int input_size_;
    int output_size_;
    Eigen::VectorXf output_;
    std::shared_ptr<Eigen::MatrixXf> weights_;
    std::shared_ptr<Eigen::MatrixXf> biases_;
    std::function<float(float)> activation_;
    std::string activation_name_;
  };

  class NeuralNetwork
  {
  public:
    NeuralNetwork() {} // ?? need default constructor that can be called by CGFEstimation default constructor?

    // ??: want vector of shared pointers instead? not sure what Eigen does internally in terms of copying/assignment
    void
    setWeightsAndBiases(const std::vector<shared_ptr<Eigen::MatrixXf>>& weights, const std::vector<shared_ptr<Eigen::MatrixXf>>& biases)
    {
      std::cout << "set\n";
      if (weights.size() != biases.size()) throw std::invalid_argument("Must have same number of weight matrices as bias vectors");
      num_layers_ = weights.size();
      for (int i = 0; i < num_layers_; i++)
      {
        layers_.push_back(Layer(weights[i], biases[i]));
        layer_sizes_.push_back(layers_.back().outputSize());
        if (i > 0 && layers_[i - 1].outputSize() != layers_[i].inputSize()) // LATER: move this code block elsewhere, eg through custom error type?
        {
          std::ostringstream err_stream;
          err_stream << "Output size of layer " << i - 1 << " (" << layers_[i - 1].outputSize() << " )";
          err_stream << " does not match input size of layer " << i << " (" << layers_[i].inputSize() << " )";
          throw std::invalid_argument(err_stream.str());
        }
      }
      input_size_ = layers_.front().inputSize();
      output_size_ = layers_.back().outputSize();
    }

    void
      applyNN(Eigen::VectorXf& input, Eigen::VectorXf& output) //?? doesn't actually modify input, not sure how to specify as const
    {
      const Eigen::VectorXf* temp = &input; // use copy assignment instead?
      for (int i = 0; i < num_layers_; i++)
      {
        layers_[i].applyLayer(*temp); // ?? idk what I'm doing
        temp = layers_[i].getOutputPtr(); // ?? shared ptr instead? 
      }
      output = *temp;
    }

    int
      getOutputSize()
    {
      return output_size_;
    }

    std::vector<Layer> layers_;
  private:
    int num_layers_;
    int input_size_;
    int output_size_;
    std::vector<int> layer_sizes_; // vector of output sizes of each layer
    
  };

  template<typename PointInT, typename PointOutT>
  class CGFEstimation : public Feature<PointInT, PointOutT>
  {
  public:
    using Ptr = shared_ptr<CGFEstimation<PointInT, PointOutT> >;
    using ConstPtr = shared_ptr<const CGFEstimation<PointInT, PointOutT> >;
    using Feature<PointInT, PointOutT>::feature_name_;
    using Feature<PointInT, PointOutT>::getClassName;
    using Feature<PointInT, PointOutT>::indices_;
    using Feature<PointInT, PointOutT>::search_radius_;
    // using Feature<PointInT, PointOutT>::search_parameter_;
    using Feature<PointInT, PointOutT>::input_;
    using Feature<PointInT, PointOutT>::surface_;

    using PointCloudIn = pcl::PointCloud<PointInT>;
    using PointCloudInPtr = typename PointCloudIn::Ptr;
    using PointCloudOut = typename Feature<PointInT, PointOutT>::PointCloudOut;
    using MatPtr = shared_ptr<Eigen::MatrixXf>;
    //////////////////////////////////// Constructors ////////////////////////////////////
    // LATER: make SphericalHistogram its own class?

    /** \brief Empty constructor. */
    CGFEstimation (): az_div_ (0), // ?? set members to zero so can check that they're properly set later?
                      el_div_ (0), // TODO: add checks to ensure CGFEstimation has been properly initialized
                      rad_div_ (0),
                      rmin_ (0.0),
                      rmax_ (0.0),
                      rRF_ (0.0)
    {} // ?? need empty constructor for PCL API?

    CGFEstimation(int az_div, int el_div, int rad_div)
    {
      // Histogram stuff:
      az_div_ = az_div;
      el_div_ = el_div;
      rad_div_ = rad_div;
      radiusThresholds();

      rmin_ = .1;
      rmax_ = 1.2;
      rRF_ = .25;
      // search_radius_ = rmax_; // ?? not sure I'm using inheritance properly for search stuff
      int N = az_div * el_div * rad_div;
      sph_hist_ = Eigen::VectorXf::Zero(N);

      //Other: 
      feature_name_ = "CGFEstimation";
    }

    void
      readMatrices(std::vector<MatPtr>& weights, std::vector<MatPtr>& biases, std::string file_str);
      // readMatrices(std::vector<Eigen::MatrixXf>& weights, std::vector<Eigen::MatrixXf>& biases, std::string file_str);

    //////////////////////////////////// Getters and Setters ////////////////////////////////////

    inline void
      getHistogramDivisions(int &az_div, int &el_div, int &rad_div)
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
      // std::vector<Eigen::MatrixXf> weights;
      // std::vector<Eigen::MatrixXf> biases;
      std::vector<MatPtr> weights;
      std::vector<MatPtr> biases;
      readMatrices(weights, biases, file_str);

      compression_.setWeightsAndBiases(weights, biases);
      std::cout << "finished setting compression \n" ;
      if (compression_.getOutputSize() != PointOutT::descriptorSize())
      {
        std::ostringstream err_stream;
        err_stream << "Output size of neural network ( " << compression_.getOutputSize() << " )";
        err_stream << " does not match dimensionality of feature ( " << PointOutT::descriptorSize() << " )";
        throw std::invalid_argument(err_stream.str());
      }
    }

    NeuralNetwork
      getCompression()
      {
        return compression_;
      }

  protected:

    //////////////////////////////////// Local Reference Functions ////////////////////////////////////

    unsigned int
      computeWeightedCovarianceMatrix(
      const pcl::PointCloud<PointInT>& cloud,
      const Indices& indices,
      Eigen::Matrix3f& covariance_matrix,
      const Eigen::VectorXf& weights);

    void
      disambiguateRF();

    void
      localRF(PointCloud<PointInT>& nn_cloud, std::vector<int> nn_indices_RF, std::vector<float>& nn_dists_RF, float radius);

    //////////////////////////////////// Histogram Functions ////////////////////////////////////
    inline float
      azimuth(PointInT pt);

    inline float
      elevation(PointInT pt);

    inline float
      radius(PointInT pt);

    void
      radiusThresholds();

    int
      threshold(float val, float divs);

    int
      threshold(float val, const Eigen::VectorXf thresholds_vec);

    int
      getBin(PointInT pt);

    void
      computeSphericalHistogram(const PointInT& pt, PointCloud<PointInT>& nn_cloud);

    //////////////////////////////////// Feature Computation ////////////////////////////////////

    void
      computeCGFSignature(const PointInT& pt, PointCloud<PointInT>& nn_cloud);

    /** \brief Estimate the set of all CGF (Compact Geometric Feature) signatures for the input cloud
      *
      */
    void
      computeCGFSignatures();

    void
      computeFeature(PointCloudOut& output) override;


    //////////////////////////////////// Member Variables ////////////////////////////////////

    // Histogram parameters:
    uint az_div_, el_div_, rad_div_;
    Eigen::VectorXf rad_thresholds_;
    // LATER: make setter for rRF_, rmin_, rmax_
    float rmin_, rmax_; // bin range for radius, rmax_ sets nearest neighbor search range for histogram, paper: [1.5%, 17%] or [.1m, 1.2m] // TODO: set
    float rRF_; // nearest neighbor radius for LRF generation, paper: 2% of model or .25 m for LIDAR
    // TODO: add check for rRF_ < rmax_
    Eigen::VectorXf sph_hist_;

    // Nearest neighbor temp storage
    std::vector<int> nn_indices_;
    std::vector<float> nn_dists_;
    std::vector<int> nn_indices_RF_;
    std::vector<float> nn_dists_RF_;
    PointCloudIn nn_cloud_;

    // Point cloud transformation and local RF stuff
    Eigen::Matrix3f cov_;
    Eigen::Matrix3f eig_;
    Eigen::Affine3f transformation_;
    
    // Neural network compression and results temp storage
    pcl::NeuralNetwork compression_;
    Eigen::VectorXf signature_;
    PointCloudOut output_;
  };
}

