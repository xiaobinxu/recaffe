#ifndef caffeine_BLOB_HPP
#define caffeine_BLOB_HPP

#include <memory>
#include "caffeine/syncedmem.hpp"
#include "caffeine/common.hpp"

namespace caffeine {

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
    void check_data();
    void check_diff();
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> diff_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;
};  // class Blob


}  // namespace caffeine

#endif  // caffeine_BLOB_HPP
