#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::Forward(vector<const Blob<Dtype>*>& bottom,
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
void Layer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom,
    vector<const Blob<Dtype>*>* top, bool propagate_down) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Backward_cpu(bottom, top, propagate_down);
      break;
    case Caffe::GPU:
      Backward_gpu(bottom, top, propagate_down);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}


}  // namespace caffe
