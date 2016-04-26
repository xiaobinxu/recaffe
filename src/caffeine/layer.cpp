#include "caffeine/layer.hpp"

namespace caffeine {

template <typename Dtype>
void Layer<Dtype>::Forward(vector<const Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffeine::mode()) {
    case Caffeine::CPU:
      Forward_cpu(bottom, top);
      break;
    case Caffeine::GPU:
      Forward_gpu(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown caffeine mode.";
  }
}

template <typename Dtype>
void Layer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom,
    vector<const Blob<Dtype>*>* top, bool propagate_down) {
  switch (Caffeine::mode()) {
    case Caffeine::CPU:
      Backward_cpu(bottom, top, propagate_down);
      break;
    case Caffeine::GPU:
      Backward_gpu(bottom, top, propagate_down);
      break;
    default:
      LOG(FATAL) << "Unknown caffeine mode.";
  }
}


}  // namespace caffeine
