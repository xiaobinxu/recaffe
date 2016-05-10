#ifndef CAFFE_VISION_LAYERS_HPP
#define CAFFE_VISION_LAYERS_HPP

#include <leveldb/db.h>
#include <pthread.h>

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
  public:
    explicit NeuronLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>* top);
};

template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
  public:
    explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);

    virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down, vector<Blob<Dtype>*>* bottom);
    virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
  public:
    explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
        vector<Blob<Dtype>*>* top);

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        vector<Blob<Dtype>*>* top);

    virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down, vector<Blob<Dtype>*>* bottom);
    virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
        bool propagate_down, vector<Blob<Dtype>*>* bottom);

    int M_;
    int K_;
    int N_;
    bool biasterm_;
    shared_ptr<SyncedMemory> bias_multiplier_;
};

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* DataLayerPrefetch<Dtype>(void*);

 public:
  explicit DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
};

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);

  // sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  // scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
};

template <typename Dtype>
class MultinomialLogisticLossLayer : public Layer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


// SoftmaxWithLossLayer is a layer that implements softmax and then computes
// the loss - it is preferred over softmax + multinomiallogisticloss in the
// sense that during training, this will produce more numerically stable
// gradients. During testing this layer could be replaced by a softmax layer
// to generate probability outputs.
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};

template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // The accuracy layer should not be used to compute backward operations.
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
    return Dtype(0.);
  }
};

}  // namespace caffe


#endif  // CAFFE_VISION_LAYERS_HPP
