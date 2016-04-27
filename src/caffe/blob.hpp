#ifndef CAFFE_BLOB_HPP
#define CAFFE_BLOB_HPP

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/layer_param.pb.h"

namespace caffe {

template <typename Dtype>
class Blob
{
  public:
    Blob()
      : num_(0), channels_(0), height_(0), width_(0), count_(0), data_(),
      diff_() {} 
    explicit Blob(int num, int channels, int height, int width)
    {
      Reshape(num, channels, height, width);
    }
    ~Blob() {}
    void Reshape(int num, int channels, int height, int width);
    int num() { return num_; }
    int channels() { return channels_; }
    int height() { return height_; }
    int width() { return width_; }
    int count() { return count_; }

    const Dtype* cpu_data();
    const Dtype* gpu_data();
    const Dtype* cpu_diff();
    const Dtype* gpu_diff();
    Dtype* mutable_cpu_data();
    Dtype* mutable_gpu_data();
    Dtype* mutable_cpu_diff();
    Dtype* mutable_gpu_diff();
    void Update();
    void FromProto(const BlobProto& proto);
    void ToProto(BlobProto* proto);

  private:
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> diff_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;
};  // class Blob


}  // namespace caffe

#endif  // CAFFE_BLOB_HPP
