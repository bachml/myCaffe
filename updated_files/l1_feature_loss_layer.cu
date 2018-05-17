#include <vector>

#include "caffe/layers/l1_feature_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void shostakovich_sym_no_5(const int count, const Dtype* bottom_data, Dtype* diff_data) {
    CUDA_KERNEL_LOOP(index, count) {
        if (bottom_data[index] == 0) {
            diff_data[index] = 0;
        }
    }
}




template <typename Dtype>
void L1FeatureLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //caffe_gpu_sub(
  //    count,
  //    bottom[0]->gpu_data(),
  //    bottom[1]->gpu_data(),
  //    diff_.mutable_gpu_data());
  
  //caffe_copy(count, bottom[0]->gpu_data(), diff_.mutable_gpu_data());
  //Dtype dot;
  //caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //top[0]->mutable_cpu_data()[0] = loss;


  caffe_gpu_abs(count, bottom[0]->gpu_data(), temp_abs_.mutable_gpu_data());
  caffe_gpu_div(count, bottom[0]->gpu_data(), temp_abs_.gpu_data(), diff_.mutable_gpu_data());
  

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* diff_data = diff_.mutable_gpu_data();

  //for (int i=0; i < bottom[0]->count(); i++) {
  //  if ((bottom_data[i]) == 0) {
  //    diff_data[i] = 0;   
  //  }
  //}
  shostakovich_sym_no_5<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, diff_data);
  CUDA_POST_KERNEL_CHECK;



  Dtype sum;
  //caffe_gpu_asum(count, temp_abs_.gpu_data(), &sum);
  caffe_gpu_dot(count, temp_abs_.gpu_data(), uni_temp_.gpu_data(), &sum);

  Dtype loss = sum / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L1FeatureLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /*for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }*/
    const Dtype sign = 1;
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_gpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_gpu_diff());  // b
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(L1FeatureLossLayer);

}  // namespace caffe
