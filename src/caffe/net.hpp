#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::vector;
using std::string;

namespace caffe {

template <typename Dtype>
class Net {
  public:
    Net(const NetParameter& param, const vector<Blob<Dtype>*>& bottom);
    ~Net() {}

    const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>*>& bottom);
    // The network backward should take no input and output, since it solely
    // computes the gradient w.r.t the parameters, and the data has already
    // been provided during the forward pass.
    Dtype Backward();

    Dtype ForwardBackward(const vector<Blob<Dtype>*>& bottom) {
      Forward(bottom);
      return Backward();
    }

    // For an already initialized net, CopyTrainedLayersFrom() copies the
    // already trained layers from another net parameter instance.
    void CopyTrainedLayersFrom(const NetParameter& param);
    // Writes the net to a proto.
    void ToProto(NetParameter* param, bool write_diff = false);

    const string& name() { return name_; }
    const vector<string>& layer_names() { return layer_names_; }
    const vector<string>& blob_names() { return blob_names_; }
    const vector<shared_ptr<Layer<Dtype> > >& layers() { return layers_; }
    const vector<shared_ptr<Blob<Dtype> > >& blobs() { return blobs_; }
    vector<shared_ptr<Blob<Dtype> > >& params() { return params_; }
    // returns the bottom and top vecs for each layer - usually you won't need
    // this unless you do per-layer checks such as gradients.
    vector<vector<Blob<Dtype>*> >& bottom_vecs() { return bottom_vecs_; }
    vector<vector<Blob<Dtype>*> >& top_vecs() { return top_vecs_; }

    // update the network
    void Update();

  protected:
    // individual layers in the net
    vector<shared_ptr<Layer<Dtype> > > layers_;
    vector<string> layer_names_;
    // stores the blobs that store intermidiate results between the layers
    vector<shared_ptr<Blob<Dtype> > > blobs_;
    vector<string> blob_names_;
    // bottom_vecs stores the vectors containing the input for each layer
    vector<vector<Blob<Dtype>*> > bottom_vecs_;
    vector<vector<int> > bottom_id_vecs_;
    // top_vecs stores the vectors containing the output for each layer
    vector<vector<Blob<Dtype>*> > top_vecs_;
    vector<vector<int> > top_id_vecs_;
    // blob indices for the input and the output of the net
    vector<int> net_input_blob_indices_;
    vector<int> net_output_blob_indices_;
    vector<Blob<Dtype>*> net_output_blobs_;

    string name_;
    vector<shared_ptr<Blob<Dtype> > > params_;

    DISABLE_COPY_AND_ASSIGN(Net);
};  // class Net

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
