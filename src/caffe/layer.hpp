#ifndef CAFFE_LAYER_HPP
#define CAFFE_LAYER_HPP

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

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
      : layer_param_(param) {
        // the only thing we do is to copy blobs if ther are any.
        if (layer_param_.blobs_size() > 0) {
          blobs_.resize(layer_param_.blobs_size());
          for (int i = 0; i < layer_param_.blobs_size(); ++i) {
            blobs_[i].reset(new Blob<Dtype>());
            blobs_[i]->FromProto(layer_param_.blobs(i));
          }
        }
    }
    virtual ~Layer() {}

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

    vector<shared_ptr<Blob<Dtype> > >& blobs() { return blobs_; }
    const LayerParameter& layer_param() { return layer_param_; }
    virtual void ToProto(LayerParameter* param, bool write_diff = false);
  
  protected:
    // The protobuf that stores the layer parameters
    LayerParameter layer_param_;
    // The vector that stores the parameters and a set of blobs
    vector<shared_ptr<Blob<Dtype> > > blobs_;

    // Forward functions
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) {
      LOG(WARNING) << "Using CPU code as backup.";
      Forward_cpu(bottom, top);
    }

    // Backward functions: the backward function will compute the gradients for
    // any parameters and also for the bottom blobs if propagate_down is true.
    // It will return the loss produced from this layer.
    virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom) = 0;
    virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom) {
      LOG(WARNING) << "Using CPU code as backup.";
      return Backward_cpu(top, propagate_down, bottom);
    }

    DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer


template <typename Dtype>
void Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Forward_cpu(bottom, top);
      break;
    case Caffe::GPU:
      Forward_gpu(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
Dtype Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      return Backward_cpu(top, propagate_down, bottom);
    case Caffe::GPU:
      return Backward_gpu(top, propagate_down, bottom);
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i)
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
}


}  // namespace caffe

#endif  // CAFFE_LAYER_HPP
