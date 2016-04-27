#include "caffe/layer.hpp"

namespace caffe {

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
      break;
    case Caffe::GPU:
      return Backward_gpu(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Layer<Dtype>::Predict(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Predict_cpu(bottom, top);
      break;
    case Caffe::GPU:
      Predict_gpu(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

}  // namespace caffe
