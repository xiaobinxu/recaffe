#include <algorithm>
#include <fstream>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/optimization/solver.hpp"

using std::max;
using std::min;
using std::stringstream;
using std::ofstream;

namespace caffe {

template <typename Dtype>
void Solver<Dtype>::Solve(Net<Dtype>* net) {
  net_ = net;
  LOG(INFO) << "Solving " << net_->name();
  PreSolve();
  iter_ = 0;
  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  while (iter_++ < param_.max_iter()) {
    Dtype loss = net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    net_->Update();

    // Check if we need to do snapshot
    if (param_.snapshot() > 0 && iter_ % param_.snapshot() == 0) {
      Snapshot(false);
    }
    if (param_.display()) {
      LOG(ERROR) << "Iteration " << iter_ << ", loss = " << loss;
    }
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::Snapshot(bool is_final) {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, !is_final);
  stringstream ss;
  ss << param_.snapshot_prefix();
  if (is_final) {
    ss << "_final";
  } else {
    ss << "_iter_" << iter_;
  }
  string filename = ss.str();
  LOG(ERROR) << "Snapshotting to " << filename;
  WriteProtoToBinaryFile(net_param, filename.c_str());
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  rate = min(max(rate, Dtype(this->param_.min_lr())),
      Dtype(this->param_.max_lr()));
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // First of all, see if we need to initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  for (size_t i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  // get the learning rate
  Dtype rate = GetLearningRate();
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  // LOG(ERROR) << "rate:" << rate << " momentum:" << momentum
  //     << " weight_decay:" << weight_decay;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (size_t param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      caffe_axpby(net_params[param_id]->count(), rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
      if (weight_decay) {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(), weight_decay * rate,
            net_params[param_id]->cpu_data(),
            history_[param_id]->mutable_cpu_data());
      }
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (size_t param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      caffe_gpu_axpby(net_params[param_id]->count(), rate,
          net_params[param_id]->gpu_diff(), momentum,
          history_[param_id]->mutable_gpu_data());
      if (weight_decay) {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(), weight_decay * rate,
            net_params[param_id]->gpu_data(),
            history_[param_id]->mutable_gpu_data());
      }
      // copy
      caffe_gpu_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);

}  // namespace caffe
