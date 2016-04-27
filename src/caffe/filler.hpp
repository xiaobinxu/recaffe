#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <mkl.h>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/layer_param.pb.h"

namespace caffe {

template <typename Dtype>
class Filler {
  public:
    Filler(const FillerParameter& param) : filler_param_(param) {}
    virtual ~Filler() {}
    virtual void Fill(Blob<Dtype>* blob) = 0;
  protected:
    FillerParameter filler_param_;
};  // class Filler

template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
  public:
    ConstantFiller(const FillerParameter& param) : Filler<Dtype>(param) {}
    virtual void Fill(Blob<Dtype>* blob)
    {
      Dtype* data = blob->mutable_cpu_data();
      const int count = blob->count();
      const Dtype value = this->filler_param_.value();
      CHECK(count);
      for (int i = 0; i < count; ++i) {
        data[i] = value;
      }
    }
};  // class ConstantFiller

template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
  public:
    UniformFiller(const FillerParameter& param) : Filler<Dtype>(param) {}
    virtual void Fill(Blob<Dtype>* blob)
    {
      void* data = (void*)(blob->mutable_cpu_data());
      const int count = blob->count();
      CHECK(count);
      switch (sizeof(Dtype)) {
        case sizeof(float):
          VSL_CHECK(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                Caffe::vsl_stream(), count, (float*)data,
                this->filler_param_.min(), this->filler_param_.max()));
          break;
        case sizeof(double):
          VSL_CHECK(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                Caffe::vsl_stream(), count, (double*)data,
                this->filler_param_.min(), this->filler_param_.max()));
          break;
        default:
          CHECK(false) << "Unknown dtype.";
      }
    }
};  // class UniformFiller

template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
  public:
    GaussianFiller(const FillerParameter& param) : Filler<Dtype>(param) {}
    virtual void Fill(Blob<Dtype>* blob)
    {
      void* data = (void*)(blob->mutable_cpu_data());
      const int count = blob->count();
      CHECK(count);
      switch (sizeof(Dtype)) {
        case sizeof(float):
          VSL_CHECK(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
                Caffe::vsl_stream(), count, (float*)data,
                this->filler_param_.mean(), this->filler_param_.std()));
          break;
        case sizeof(double):
          VSL_CHECK(vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
                Caffe::vsl_stream(), count, (double*)data,
                this->filler_param_.mean(), this->filler_param_.std()));
          break;
        default:
          CHECK(false) << "Unknown dtype.";
      }
    }
};  // class GaussianFiller


}  // namespace caffe


#endif  // CAFFE_FILLER_HPP
