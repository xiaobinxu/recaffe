#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_vRngUniform<float>(const int n, float* r, const float a, const float b) {
  VSL_CHECK(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
        Caffe::vsl_stream(), n, r, a, b));
}

template <>
void caffe_vRngUniform<double>(const int n, double* r, const double a, const double b) {
  VSL_CHECK(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
        Caffe::vsl_stream(), n, r, a, b));
}

template <>
void caffe_vRngGaussian<float>(const int n, float* r, const float a, const float sigma) {
  VSL_CHECK(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
        Caffe::vsl_stream(), n, r, a, sigma));
}

template <>
void caffe_vRngGaussian<double>(const int n, double* r, const double a, const double sigma) {
  VSL_CHECK(vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
        Caffe::vsl_stream(), n, r, a, sigma));
}

}  // namespace caffe
