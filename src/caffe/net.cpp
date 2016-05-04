#include <map>
#include <set>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"

using std::pair;
using std::map;
using std::set;

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const vector<Blob<Dtype>*>& bottom) {
  // basically, build all the layers and set up its connections
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  CHECK_EQ(bottom.size(), param.input_size()) << "Incorrect bottom blob size.";
  // set the input blobs
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    CHECK_GT(bottom[i]->count(), 0);
    shared_ptr<Blob<Dtype> > blob_pointer(
        new Blob<Dtype>(bottom[i]->num(), bottom[i]->channels(),
          bottom[i]->height(), bottom[i]->width()));
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    net_input_blob_indices_.push_back(i);
    blob_name_to_idx[blob_name] = i;
    available_blobs.insert(blob_name);
  }
  // for each layer, set up their input and output
  bottom_vecs_.resize(param.layers_size());
  bottom_id_vecs_.resize(param.layers_size());
  top_vecs_.resize(param.layers_size());
  top_id_vecs_.resize(param.layers_size());
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerConnection& layer_connection = param.layers(i);
    const LayerParameter& layer_param = layer_connection.layer();
    layers_.push_back(shared_ptr<Layer<Dtype> >(GetLayer<Dtype>(layer_param)));
    layer_names_.push_back(layer_param.name());
    LOG(INFO) << "Creating layer " << layer_param.name();
    // figure out this layer's input and output
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      if (available_blobs.find(blob_name) == available_blobs.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name << " to layer" << j;
      }
      LOG(INFO) << layer_param.name() << " <- " << blob_name;
      bottom_vecs_[i].push_back(blobs_[blob_name_to_idx[blob_name]].get());
      bottom_id_vecs_[i].push_back(blob_name_to_idx[blob_name]);
      available_blobs.erase(blob_name);
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      if (blob_name_to_idx.find(blob_name) != blob_name_to_idx.end()) {
        LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
      }
      LOG(INFO) << layer_param.name() << " -> " << blob_name;
      shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
      blobs_.push_back(blob_pointer);
      blob_names_.push_back(blob_name);
      blob_name_to_idx[blob_name] = blob_names_.size() - 1;
      available_blobs.insert(blob_name);
      top_vecs_[i].push_back(blobs_[blob_names_.size() - 1].get());
      top_id_vecs_[i].push_back(blob_names_.size() - 1);
    }
  }
  // in the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG(ERROR) << "This network produces output " << *it;
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
  }

  LOG(ERROR) << "Setting up the layers.";
  for (int i = 0; i < layers_.size(); ++i) {
    LOG(INFO) << "Setting up " << layer_names_[i];
    layers_[i]->SetUp(bottom_vecs_[i], &top_vecs_[i]);
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = layers_[i]->blobs();
    for (int j = 0; j < layer_blobs.size(); ++j) {
      params_.push_back(layer_blobs[j]);
    }
    for (int topid = 0; topid < top_vecs_[i].size(); ++topid) {
      LOG(INFO) << "Top shape: " << top_vecs_[i][topid]->channels() << " "
        << top_vecs_[i][topid]->height() << " "
        << top_vecs_[i][topid]->width();
    }
  }

  LOG(ERROR) << "Network initialization done.";
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom) {

  // copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    blobs_[net_input_blob_indices_[i]]->CopyFrom(*bottom[i]);
  }
  for (int i = 0; i < layers_.size(); ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    layers_[i]->Forward(bottom_vecs_[i], &top_vecs_[i]);
  }
  return net_output_blobs_;
}

template <typename Dtype>
Dtype Net<Dtype>::Backward() {
  Dtype loss = 0;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    Dtype layer_loss = layers_[i]->Backward(
        top_vecs_[i], true, &bottom_vecs_[i]);
    loss += layer_loss;
  }
  return loss;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layers_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layers(i).layer();
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    LOG(INFO) << "Loading source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
      layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
      << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      CHECK_EQ(target_blobs[j]->num(), source_layer.blobs(j).num());
      CHECK_EQ(target_blobs[j]->channels(), source_layer.blobs(j).channels());
      CHECK_EQ(target_blobs[j]->height(), source_layer.blobs(j).height());
      CHECK_EQ(target_blobs[j]->width(), source_layer.blobs(j).width());
      target_blobs[j]->FromProto(source_layer.blobs(j));
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) {
  param->Clear();
  param->set_name(name_);
  // add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_input(blob_names_[net_input_blob_indices_[i]]);
  }
  LOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerConnection* layer_connection = param->add_layers();
    for (int j = 0; j < bottom_id_vecs_[i].size(); ++j) {
      layer_connection->add_bottom(blob_names_[bottom_id_vecs_[i][j]]);
    }
    for (int j = 0; j < top_id_vecs_[i].size(); ++j) {
      layer_connection->add_top(blob_names_[top_id_vecs_[i][j]]);
    }
    LayerParameter* layer_parameter = layer_connection->mutable_layer();
    layers_[i]->ToProto(layer_parameter, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < params_.size(); ++i)
    params_[i]->Update();
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
