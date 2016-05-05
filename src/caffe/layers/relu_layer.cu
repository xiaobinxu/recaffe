
#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
