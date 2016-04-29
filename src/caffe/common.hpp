#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <cstddef>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <mkl_vsl.h>

#include <cublas_v2.h>
#include <driver_types.h>

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;
using std::size_t;

#define CUDA_CHECK(condition) CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition) CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define VSL_CHECK(condition) CHECK_EQ((condition), VSL_STATUS_OK)

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas.
class Caffe
{
  public:
    ~Caffe();
    static Caffe& Get();
    enum Brew { CPU, GPU };

    // The getters for the variables
    static cublasHandle_t cublas_handle();
    static VSLStreamStatePtr vsl_stream();
    static Brew mode();
    // The setters for the variables
    static void set_mode(Brew mode);

  private:
    Caffe();
    static shared_ptr<Caffe> singleton_;
    cublasHandle_t cublas_handle_;
    VSLStreamStatePtr vsl_stream_;
    Brew mode_;
};


}  // namespace caffe


#endif  // CAFFE_COMMOM_HPP_
