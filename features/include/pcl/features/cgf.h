
#pragma once
#define PCL_NO_PRECOMPILE
#include <pcl/features/feature.h>
#include <pcl/common/impl/io.hpp> // need for copyPointCloud()
#include <pcl/common/impl/transforms.hpp> // need for transformPointCloud()
#include <pcl/common/centroid.h> // demeanPointCloud()
#include <pcl/common/eigen.h> // eigen33()
#include <pcl/common/distances.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>

namespace pcl
{
  template <typename FeatureType>
  struct DescriptorRepresentation : public PointRepresentation<FeatureType> // LATER: best practices?
      {
        // public:
      DescriptorRepresentation ()
      {
        nr_dimensions_ = FeatureType::descriptorSize;
      }
      ~DescriptorRepresentation () {}

      void
      copyToFloatArray (const FeatureType &p, float * out) const override
      {
        for (int i = 0; i < nr_dimensions_; ++i)
          out[i] = p.descriptor[i];
      }
      };

  class Layer
  {
  public:

    Layer(const shared_ptr<Eigen::MatrixXf>& weights, const shared_ptr<Eigen::MatrixXf>& biases)
    {
      std::cout<< "layer constructor\n";
      setWeightsAndBiases(weights, biases);
      set_activation_relu(1.0); // LATER: make possible to set from constructor
      std::cout << "output_size_: " << output_size_ << "\n" ;
      Eigen::VectorXf output = Eigen::VectorXf::Zero(output_size_);
      output_ptr_ = make_shared<Eigen::VectorXf>(output);
    }

    void
      setWeightsAndBiases(const shared_ptr<Eigen::MatrixXf>& weights, const shared_ptr<Eigen::MatrixXf>& biases)
    {
      // LATER: add check for bias being column vectors
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
      weights = *weights_; 
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
      applyLayer(const Eigen::VectorXf& input)
    {
      *output_ptr_ = (*weights_ * input + *biases_).unaryExpr(std::ref(activation_));
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
    // Eigen::VectorXf*
    shared_ptr<Eigen::VectorXf>
      getOutputPtr()
    {
      return output_ptr_; // ?? return the same pointer or move/copy to new pointer/pointer passed by ref?
    }

  private:
    int input_size_;
    int output_size_;
    std::shared_ptr<Eigen::VectorXf> output_ptr_;
    std::shared_ptr<Eigen::MatrixXf> weights_;
    std::shared_ptr<Eigen::MatrixXf> biases_;
    std::function<float(float)> activation_;
    std::string activation_name_;
  };

  class NeuralNetwork
  {
  public:
    NeuralNetwork() : initialized_ (false) {} // default constructor that can be called by CGFEstimation default constructor 

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
          // std::ostringstream err_stream;
          // err_stream << "Output size of layer " << i - 1 << " (" << layers_[i - 1].outputSize() << " )";
          // err_stream << " does not match input size of layer " << i << " (" << layers_[i].inputSize() << " )";
          // throw std::invalid_argument(err_stream.str());
          throw DimensionMismatch(layers_, i);
        }
      }
      input_size_ = layers_.front().inputSize();
      output_size_ = layers_.back().outputSize();
      initialized_ = true;
    }

    void
      applyNN(Eigen::VectorXf& input, Eigen::VectorXf& output) //?? doesn't actually modify input, not sure how to specify as const
    {
      // const Eigen::VectorXf* temp = &input; // use copy assignment instead?
      shared_ptr<Eigen::VectorXf> temp = make_shared<Eigen::VectorXf> (input); // use copy assignment instead?
      for (int i = 0; i < num_layers_; i++)
      {
        layers_[i].applyLayer(*temp); // ?? idk what I'm doing
        temp = move(layers_[i].getOutputPtr()); // ?? shared ptr instead? 
      }
      output = *temp;
    }

    int
      getOutputSize()
    {
      return output_size_;
    }

    bool 
      initialized()
      {
        return initialized_;
      }

    class DimensionMismatch : public std::invalid_argument { // ?? is this the "right" way to do this? lol
      public:
        DimensionMismatch() : std::invalid_argument("Layer dimension mismatch") { }
        DimensionMismatch(std::vector<Layer>& layers, int i) :
        std::invalid_argument(makeErrorString(layers, i)) { }

      private:
        std::string
          makeErrorString(std::vector<Layer>& layers, int i)
          {
          std::ostringstream err_stream;
          err_stream << "Output size of layer " << i - 1 << " (" << layers[i - 1].outputSize() << " )";
          err_stream << " does not match input size of layer " << i << " (" << layers[i].inputSize() << " )";
          return err_stream.str();
          }
    };

    std::vector<Layer> layers_;
  private:
    bool initialized_; // ?? restructure to use nullptr in CGFEstimation constructor for compression_ member and change NeuralNetwork constructor to take args?
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
    using Feature<PointInT, PointOutT>::search_parameter_;
    using Feature<PointInT, PointOutT>::input_;
    using Feature<PointInT, PointOutT>::surface_;

    using PointCloudIn = pcl::PointCloud<PointInT>;
    using PointCloudInPtr = typename PointCloudIn::Ptr;
    using PointCloudOut = typename Feature<PointInT, PointOutT>::PointCloudOut;
    using MatPtr = shared_ptr<Eigen::MatrixXf>;
    //////////////////////////////////// Constructors ////////////////////////////////////
    // LATER: make SphericalHistogram its own class?

    // /** \brief Empty constructor. */
    // CGFEstimation (): az_div_ (0), // ?? set members to zero so can check that they're properly set later?
    //                   el_div_ (0), // TODO: add checks to ensure CGFEstimation has been properly initialized
    //                   rad_div_ (0),
    //                   rmin_ (0.0),
    //                   rmax_ (0.0),
    //                   rRF_ (0.0)
    // {}

    CGFEstimation(int az_div, int el_div, int rad_div, float rmin =.1, float rmax = 1.2, float rRF = .25)
    {
      // Histogram stuff:
      az_div_ = az_div;
      el_div_ = el_div;
      rad_div_ = rad_div;
      rmin_ = rmin;
      rmax_ = rmax;
      radiusThresholds();

      rRF_ = rRF;
      this -> setRadiusSearch(rmax_); 

      int N = az_div * el_div * rad_div;
      sph_hist_ = Eigen::VectorXf::Zero(N);

      feature_name_ = "CGFEstimation";
      descriptor_size_ = PointOutType::descriptorSize();
    }

    void
      readMatrices(std::vector<MatPtr>& weights, std::vector<MatPtr>& biases, std::string file_str);

    template <typename T, size_t N> void
      print_arr(T (&arr)[N]);

    //////////////////////////////////// Exceptions ////////////////////////////////////

    class UninitializedException : public std::logic_error { // ?? is this the "right" way to do this? lol
      public:
        UninitializedException() : std::logic_error("Uninitialized exception") { }
        UninitializedException(std::string str) : std::logic_error(str) { }
    };

    //////////////////////////////////// Getters and Setters ////////////////////////////////////
    // =====GETTERS=====
      //! Getter (const) for the descriptor
    // inline const float* 
    //   getDescriptor () const { return descriptor_;}
    // //! Getter for the descriptor
    // inline float* 
    //   getDescriptor () { return descriptor_;}

    //! Getter for the descriptor length
    inline int //?? why did example return reference?
      getDescriptorSize () { return descriptor_size_;}
    // //! Getter (const) for the position
    // inline const Eigen::Vector3f& 
    //   getPosition () const { return position_;}
    // //! Getter for the position
    // inline Eigen::Vector3f& 
    //   getPosition () { return position_;}
    // //! Getter (const) for the 6DoF pose
    // inline const Eigen::Affine3f& 
    //   getTransformation () const { return transformation_;}
    // //! Getter for the 6DoF pose
    // inline Eigen::Affine3f& 
    //   getTransformation () { return transformation_;}

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
      std::vector<MatPtr> weights;
      std::vector<MatPtr> biases;
      readMatrices(weights, biases, file_str);

      compression_.setWeightsAndBiases(weights, biases);
      std::cout << "finished setting compression \n" ;
      if (compression_.getOutputSize() != descriptor_size_)
      {
        std::ostringstream err_stream;
        err_stream << "Output size of neural network ( " << compression_.getOutputSize() << " )";
        err_stream << " does not match dimensionality of feature ( " << descriptor_size_ << " )";
        throw std::invalid_argument(err_stream.str());
      }
    }

    NeuralNetwork
      getCompression()
      {
        return compression_;
      }

  // protected: // TEMP
    // TODO: change member functions to primarily act on member variables rather than inputs
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
      localRF();

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

    int 
      ORsearchForNeighbors(int idx, float radius, std::vector<int>& nn_indices, std::vector<float>& nn_dists); // TEMP fn for externally checking nearest neighbor search

    void
      computeSphericalHistogram(const PointInT& pt);

    //////////////////////////////////// Feature Computation ////////////////////////////////////

    void
      computeCGFSignature(const PointInT& pt);

    /** \brief Estimate the set of all CGF (Compact Geometric Feature) signatures for the input cloud
      *
      */
    void
      // computeCGFSignatures();
      computeCGFSignatures(PointCloudOut& output);

    void
      computeFeature(PointCloudOut& output) override;

    //////////////////////////////////// Member Variables ////////////////////////////////////

    // Histogram parameters:
    uint az_div_, el_div_, rad_div_;
    Eigen::VectorXf rad_thresholds_;
    uint descriptor_size_;
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
    Eigen::Matrix3f lrf_;
    Eigen::Affine3f transformation_;
    
    // Neural network compression and results temp storage
    pcl::NeuralNetwork compression_;
    Eigen::VectorXf signature_;
  };
}
