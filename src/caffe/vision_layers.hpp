#ifndef CAFFE_VISION_LAYERS_HPP
#define CAFFE_VISION_LAYERS_HPP

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
  public:
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>* top);
};

template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);

    virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom);
    virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down,
        vector<Blob<Dtype>*>* bottom);

    virtual void Predict_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
    virtual void Predict_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
};


}  // namespace caffe


#endif  // CAFFE_VISION_LAYERS_HPP
