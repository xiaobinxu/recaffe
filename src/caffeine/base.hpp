#ifndef CAFFEINE_BASE_HPP
#define CAFFEINE_BASE_HPP

#include <vector>
#include "caffeine/blob.hpp"
#include "caffeine/proto/layer_param.pb.h"

using std::vector;

namespace caffeine
{

template <typename Dtype>
class Layer
{
  public:
    explicit Layer(const LayerParameter& param)
      : initialized_(false), layer_param_(param) {}
    ~Layer();
    virtual void SetUp(vector<const Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
    virtual void Forward(vector<const Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
    virtual void Predict(vector<const Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top) = 0;
    virtual void Backward(vector<const Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top, bool propagate_down) = 0;
  
  protected:
    bool initialized_;
    LayerParameter layer_param_;
    vector<Blob<Dtype> > blobs;
};  // class Layer


}  // namespace caffeine

#endif  // CAFFEINE_BASE_HPP
