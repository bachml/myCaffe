#include <vector>

#include "caffe/layers/l1_feature_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L1FeatureLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  //    << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  temp_abs_.ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void L1FeatureLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //caffe_sub(
  //    count,
  //    bottom[0]->cpu_data(),
  //    bottom[1]->cpu_data(),
  //    diff_.mutable_cpu_data());

  //caffe_copy(count, bottom[0]->cpu_data(), diff_.mutable_cpu_data());
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //top[0]->mutable_cpu_data()[0] = loss;
  caffe_abs(count, bottom[0]->cpu_data(), temp_abs_.mutable_cpu_data());
  caffe_div(count, bottom[0]->cpu_data(), temp_abs_.cpu_data(), diff_.mutable_cpu_data());

  Dtype sum = caffe_cpu_asum(count, temp_abs_.cpu_data());
  Dtype loss = sum / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
  
  
}

template <typename Dtype>
void L1FeatureLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /*for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }*/
    const Dtype sign = 1;
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_cpu_diff());  // b



    caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());

}

#ifdef CPU_ONLY
STUB_GPU(L1FeatureLossLayer);
#endif

INSTANTIATE_CLASS(L1FeatureLossLayer);
REGISTER_LAYER_CLASS(L1FeatureLoss);

}  // namespace caffe
