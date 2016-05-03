#ifndef CAFFE_BLOB_HPP
#define CAFFE_BLOB_HPP

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"

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
    virtual ~Blob() {}
    void Reshape(int num, int channels, int height, int width);
    int num() const { return num_; }
    int channels() const { return channels_; }
    int height() const { return height_; }
    int width() const { return width_; }
    int count() const { return count_; }
    int offset(int n, int c=0, int h=0, int w=0) const {
      return ((n * channels_ + c) * height_ + h) * width_ + w;
    }

    // copy from source. If copy_diff is false, we copy the data;
    // if copy_diff is true, we copy the diff.
    void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
        bool reshape = false);

    Dtype data_at(int n, int c, int h, int w) const {
      return *(cpu_data() + offset(n, c, h, w));
    }
    Dtype diff_at(int n, int c, int h, int w) const {
      return *(cpu_diff() + offset(n, c, h, w));
    }

    const Dtype* cpu_data() const;
    const Dtype* gpu_data() const;
    const Dtype* cpu_diff() const;
    const Dtype* gpu_diff() const;
    Dtype* mutable_cpu_data();
    Dtype* mutable_gpu_data();
    Dtype* mutable_cpu_diff();
    Dtype* mutable_gpu_diff();
    void Update();
    void FromProto(const BlobProto& proto);
    void ToProto(BlobProto* proto, bool write_diff = false) const;

  private:
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> diff_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;

    DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob


}  // namespace caffe

#endif  // CAFFE_BLOB_HPP
