#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {
  
template <typename Dtype>
class NeuronLayerTest : public ::testing::Test {
 protected:
  NeuronLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>(2, 3, 4, 5)) {
    // fill the values
    
  };
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NeuronLayerTest, Dtypes);

}
