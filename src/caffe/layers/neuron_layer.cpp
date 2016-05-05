#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Neuron layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Neuron layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(),
      bottom[0]->width());
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
