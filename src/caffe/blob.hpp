#ifndef caffe_BLOB_HPP
#define caffe_BLOB_HPP

#include <memory>
#include <cublas_v2.h>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

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
    void update();

  private:
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> diff_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;
};  // class Blob



template <typename Dtype>
void Blob<Dtype>::Reshape(int num, int channels, int height, int width)
{
  CHECK_GT(num, 0);
  CHECK_GT(channels, 0);
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;
  data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
  diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data()
{
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data()
{
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff()
{
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff()
{
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data()
{
  CHECK(data_);
  return (Dtype*)data_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data()
{
  CHECK(data_);
  return (Dtype*)data_->mutable_gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff()
{
  CHECK(diff_);
  return (Dtype*)diff_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff()
{
  CHECK(diff_);
  return (Dtype*)diff_->mutable_gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::update()
{

}

template class Blob<float>;
template class Blob<double>;

}  // namespace caffe

#endif  // caffe_BLOB_HPP
