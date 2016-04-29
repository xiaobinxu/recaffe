#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"

namespace caffe {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, TestCublasHandler) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffe::cublas_handle());
}

TEST_F(CommonTest, TestVslStream) {
  EXPECT_TRUE(Caffe::vsl_stream());
}

TEST_F(CommonTest, TestBrewMode) {
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

}
