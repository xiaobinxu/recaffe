#include "caffe/common.hpp"

namespace caffe
{

shared_ptr<Caffe> Caffe::singleton_;

Caffe::Caffe()
  : mode_(Caffe::CPU)
{
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
}

Caffe::~Caffe()
{
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (vsl_stream_)
    VSL_CHECK(vslDeleteStream(&vsl_stream_));
}

Caffe& Caffe::Get()
{
  if (!singleton_)
    singleton_.reset(new Caffe());
  return *singleton_;
}

cublasHandle_t Caffe::cublas_handle()
{
  return Get().cublas_handle_;
}

VSLStreamStatePtr Caffe::vsl_stream()
{
  return Get().vsl_stream_;
}

Caffe::Brew Caffe::mode()
{
  return Get().mode_;
}

void Caffe::set_mode(Caffe::Brew mode)
{
  Get().mode_ = mode;
}


}  // namespace caffe
