#ifndef CAFFE_UTIL_MATH_FUNCTIONS_HPP
#define CAFFE_UTIL_MATH_FUNCTIONS_HPP

#include <mkl.h>
#include <cublas_v2.h>

namespace caffe {

template <typename Dtype>
void caffe_vRngUniform(const int n, Dtype* r, const Dtype a, const Dtype b);

template <typename Dtype>
void caffe_vRngGaussian(const int n, Dtype* r, const Dtype a, const Dtype sigma);


}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_HPP
