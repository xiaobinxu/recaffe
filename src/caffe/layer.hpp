#ifndef CAFFE_LAYER_HPP
#define CAFFE_LAYER_HPP

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/layer_param.pb.h"

using std::vector;

namespace caffe
{

template <typename Dtype>
class Layer
{
  public:
    // You should not implement your own constructor. Any set up code should go
    // to SetUp(), where the dimensions of the bottom blobs are provided to the
    // layer.
    explicit Layer(const LayerParameter& param)
      : layer_param_(param) {}
    virtual ~Layer();

    // SetUp: your function should implement this.
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;

    // Forward, backward and predict wrappers. You should implement the cpu and
    // gpu specific implementations instead, and should not change these
    // functions.
    void Forward(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
    Dtype Backward(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom);
    void Predict(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
  
  protected:
    // The protobuf that stores the layer parameters
    LayerParameter layer_param_;
    // The vector that stores the parameters and a set of blobs
    vector<Blob<Dtype> > blobs_;

    // Forward functions
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) {
      LOG(WARNING) << "Using CPU code as backup.";
      Forward_cpu(bottom, top);
    }

    // Backward functions
    virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom) = 0;
    virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom) {
      LOG(WARNING) << "Using CPU code as backup.";
      return Backward_cpu(top, propagate_down, bottom);
    }

    // Prediction functions: could be overridden, but the default behavior is to
    // simply call the forward functions.
    virtual void Predict_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) { Forward_cpu(bottom, top); }
    // For prediction, if there is no Predict_gpu, then there are two options:
    // to use predict_cpu as a backup, or to use forward_gpu (e.g. maybe the
    // author forgot to write what backup s/he wants?). Thus, we will require
    // the author to explicitly specify which fallback s/he wants.
    virtual void Predict_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
};  // class Layer


}  // namespace caffe

#endif  // CAFFE_LAYER_HPP
