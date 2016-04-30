#include "caffe/common.hpp"

namespace caffe
{

shared_ptr<Caffe> Caffe::singleton_;

Caffe::Caffe()
  : mode_(Caffe::CPU), phase_(Caffe::TRAIN), cublas_handle_(NULL),
  curand_generator_(NULL), vsl_stream_(NULL)
{
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  //TODO: original caffe code has bug here!
  CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, 1701ULL));
  VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
}

Caffe::~Caffe()
{
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_)
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  if (vsl_stream_)
    VSL_CHECK(vslDeleteStream(&vsl_stream_));
}

void Caffe::set_random_seed(unsigned int seed)
{
  CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, seed));
  VSL_CHECK(vslDeleteStream(&Get().vsl_stream_));
  VSL_CHECK(vslNewStream(&Get().vsl_stream_, VSL_BRNG_MT19937, seed));
}

Caffe& Caffe::Get()
{
  if (!singleton_.get())
    singleton_.reset(new Caffe());
  return *singleton_;
}

cublasHandle_t Caffe::cublas_handle()
{
  return Get().cublas_handle_;
}

curandGenerator_t Caffe::curand_generator()
{
  return Get().curand_generator_;
}

VSLStreamStatePtr Caffe::vsl_stream()
{
  return Get().vsl_stream_;
}

Caffe::Brew Caffe::mode()
{
  return Get().mode_;
}

Caffe::Phase Caffe::phase()
{
  return Get().phase_;
}

void Caffe::set_mode(Caffe::Brew mode)
{
  Get().mode_ = mode;
}

void Caffe::set_phase(Caffe::Phase phase)
{
  Get().phase_ = phase;
}

}  // namespace caffe
