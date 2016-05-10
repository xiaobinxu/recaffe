#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param)
{
  const std::string& type = param.type();
  if (type == "relu") {
    return new ReLULayer<Dtype>(param);
  } else if (type == "data") {
    return new DataLayer<Dtype>(param);
  } else if (type == "softmax") {
    return new SoftmaxLayer<Dtype>(param);
  } else if (type == "softmax_loss") {
    return new SoftmaxWithLossLayer<Dtype>(param);
  } else if (type == "multinomial_logistic_loss") {
    return new MultinomialLogisticLossLayer<Dtype>(param);
  } else if (type == "accuracy") {
    return new AccuracyLayer<Dtype>(param);
  } else if (type == "innerproduct") {
    return new InnerProductLayer<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown layer type: " << type;
  }
  // just to suppress old compiler warnings
  return (Layer<Dtype>*)NULL;
}

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_HPP_
