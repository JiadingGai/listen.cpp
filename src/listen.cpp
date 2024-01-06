#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <cfloat>
#include <omp.h>
#include <cmath>
#include "listen.h"

std::vector<float> cross_correlation(const std::vector<float> &in, const std::vector<float> &w, int stride) {
  // Only support odd-numbered kernel size larger than 3.
  assert(w.size() >= 3 && (w.size() & 0x1) != 0);
  const auto KS = w.size();
  const auto HALF = static_cast<int>((KS - 1) / 2);

  int Lin = in.size();
  // FIXME: need to support padding and dilation properly.
  int padding = 1, dilation = 1;
  int Lout = int((Lin + 2 * padding - dilation * (KS - 1) - 1) / stride + 1);
  std::vector<float> out(Lout);

  int index = 0;
  for (int i = 0; i < in.size(); i+=stride) {
    float tmp = 0.0f;
    for (int j = -HALF; j <= HALF; j++) {
      if ((i + j) >= 0 && (i + j) < in.size()) {
        tmp += in[i + j] * w[HALF + j];
      }
    }
    out[index++] = tmp;
  }

  return out;
}

void cross_correlation_c(float *out, float *in, int Lin, float *w, int KS, int stride) {
  const auto HALF = static_cast<int>((KS - 1) / 2);
  // FIXME: need to support padding and dilation properly.
  int padding = 1, dilation = 1;
  int Lout = int((Lin + 2 * padding - dilation * (KS - 1) - 1) / stride + 1);
  int index = 0;
  for (int i = 0; i < Lin; i += stride) {
    float tmp = 0.0f;
    for (int j = -HALF; j <= HALF; j++) {
      if ((i + j) >= 0 && (i + j) < Lin) {
        tmp += in[i + j] * w[HALF + j];
      }
    }
    out[index++] = tmp;
  }
}

void add_vectors_c(float *x, float *y, int L) {
  for (int i = 0; i < L; i++) {
    x[i] += y[i];
  }
}

std::vector<float> add_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    std::vector<float> result(vec1.size(), 0);

    // Check if vectors are of the same size
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors must be of the same size for addition: " << vec1.size() << " versus " << vec2.size() << std::endl;
        assert(false);
    }

    // Add corresponding elements from both vectors
    omp_set_num_threads(6);
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];

        // FIXME: no multithreading turnt on mac os?
        if (omp_get_thread_num() > 0)
          printf("Thread %u works on element %zu\n", omp_get_thread_num(), i);
    }

    return result;
}

// GELU activation function implementation
static float gelu_scalar(float x) {
  return 0.5 * x * (1.0 + std::tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

std::vector<float> gelu(const std::vector<float> &input) {
  std::vector<float> result(input.size(), 0);
  for (int i = 0; i < input.size(); i++) {
    result[i] = gelu_scalar(input[i]);
  }
  return result;
}

std::vector<float> conv1d(const std::vector<float> &in, const std::vector<int> &in_shape, const std::vector<float> &w, const std::vector<int> &w_shape, const std::vector<float> &bias, int stride) {
  // https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
  // in : [1, 80, 3000]
  // w: [384,80, 3], ks=(3,1), stride=(1,), padding=1
  // b: [384]
  // out: [1, 384, 3000]

  int N = in_shape[0];//batch size, 1
  int Cin = in_shape[1];//in channels, 80
  int Lin = in_shape[2];// input len, 3000
  assert(w_shape[0] == Cin);
  int Cout = w_shape[1];// out channels, 384
  int KS = w_shape[2]; // kernel width, 3
  //out shape = [1, 384, 3000]
  // FIXME: need to support padding and dilation properly.
  int padding = 1, dilation = 1;
  int Lout = int((Lin + 2 * padding - dilation * (KS - 1) - 1) / stride + 1);
  std::vector<float> out(N * Cout * Lout, 0);
  for (int b = 0; b < N; b++) { //batch dim
    for (int oc = 0; oc < Cout; oc++) { // out channel dim
      std::vector<float> out_tmp(Lout, 0);
      for (int k = 0; k < Cin; k++) {
        std::vector<float> w_piece(w.begin() + oc * Cin * KS + k * KS, w.begin() + oc * Cin * KS + k * KS + KS);
        std::vector<float> input_piece(in.begin() + b * Cin * Lin + k * Lin, in.begin() + b * Cin * Lin + k * Lin + Lin);
        out_tmp = add_vectors(cross_correlation(input_piece, w_piece, stride), out_tmp);
      }
      // add bias
      for (auto &x : out_tmp) {
        x += bias[oc];
      }
      //save out_tmp back to out.
      std::copy(out_tmp.begin(), out_tmp.end(), out.begin() + b * Cout * Lout + oc * Lout);
    }
  }

  return out;
}

void conv1d_c(MATRIX_T *out, MATRIX_T *in, MATRIX_T *w, MATRIX_T *bias, int stride) {
  int N = out->dim0;
  int Cin = out->dim1;
  int Lin = out->dim2;
  assert(w->dim0 == Cin);
  int Cout = w->dim1;
  int KS = w->dim2;

  // FIXME: need to support padding and dilation properly.
  int padding = 1, dilation = 1;
  int Lout = int((Lin + 2 * padding - dilation * (KS - 1) - 1) / stride + 1);

  for (int b = 0; b < N; b++) { //batch dim
    for (int oc = 0; oc < Cout; oc++) { // out channel dim
      float out_tmp[Lout];
      for (int i = 0; i < Lout; i++) {
        out_tmp[i] = 0;
      }
      for (int k = 0; k < Cin; k++) {
        float w_piece[KS];
        for (int i = 0, l = oc * Cin * KS + k * KS; l < oc * Cin * KS + k * KS + KS; i++, l++)
          w_piece[i] = Entry3D(w, oc, k, i);

        float input_piece[Lin];
        for (int i = 0, l = b * Cin * Lin + k * Lin; l < b * Cin * Lin + k * Lin + Lin; i++, l++)
          input_piece[i] = Entry3D(in, b, k, l);

        float out_cc[Lout];
        cross_correlation_c(out_cc, input_piece, Lin, w_piece, KS, stride);
        add_vectors_c(out_tmp, out_cc, Lout);

        //std::vector<float> w_piece(w.begin() + oc * Cin * KS + k * KS, w.begin() + oc * Cin * KS + k * KS + KS);
        //std::vector<float> input_piece(in.begin() + b * Cin * Lin + k * Lin, in.begin() + b * Cin * Lin + k * Lin + Lin);
        //out_tmp = add_vectors(cross_correlation(input_piece, w_piece, stride), out_tmp);
      }
      // add bias is a 1D vector of Cout
      for (int i = 0; i < Lout; i++) {
        /* std::cout << "(SimpleTheoryOfTypes) out_tmp[i] = " << out_tmp[i] << std::endl; */
        out_tmp[i] += Entry1D(bias, oc);
      }
      //save out_tmp back to out.
      //std::copy(out_tmp.begin(), out_tmp.end(), out.begin() + b * Cout * Lout + oc * Lout);
      /* std::memcpy(static_cast<void *>(&Entry3D(out, b, oc, 0)), out_tmp, Lout); */
      for (int i = 0; i < Lout; i++) {
        assert(Lout == 3000);
        Entry3D(out, b, oc, i) = out_tmp[i];
      }
    }
  }
}

MATRIX_T* Matrix_Allocate(int width, int height, int depth) {
    MATRIX_T* a_matrix = (MATRIX_T*) malloc(sizeof(MATRIX_T));
    a_matrix->dim0 = width;
    a_matrix->dim1 = height;
    a_matrix->dim2 = depth;

    a_matrix->entries = (float*) malloc(width*height*depth*sizeof(float));

    return a_matrix;
}

std::tuple<float, float, float> compare_matrix(MATRIX_T *x, MATRIX_T *y) {
  assert(x->dim0 == y->dim0);
  assert(x->dim1 == y->dim1);
  assert(x->dim2 == y->dim2);

  float max_error = FLT_MIN;
  float max_val = FLT_MIN;
  float min_val = FLT_MAX;
  for (int i = 0; i < x->dim0; i++) {
    for (int j = 0; j < x->dim1; j++) {
      for (int k = 0; k < x->dim2; k++) {
        auto curr_error = std::abs(Entry3D(x, i, j, k) - Entry3D(y, i, j, k));
        auto curr_val = Entry3D(x, i, j, k);
        max_error = std::max(curr_error, max_error);
        max_val = std::max(curr_val, max_val);
        min_val = std::min(curr_val, min_val);
      }
    }
  }

  return std::make_tuple(max_error, max_val, min_val);
}

std::tuple<float, float> compute_range(MATRIX_T *x) {
  float max_val = FLT_MIN;
  float min_val = FLT_MAX;
  for (int i = 0; i < x->dim0; i++) {
    for (int j = 0; j < x->dim1; j++) {
      for (int k = 0; k < x->dim2; k++) {
        auto curr_val = Entry3D(x, i, j, k);
        max_val = std::max(curr_val, max_val);
        min_val = std::min(curr_val, min_val);
      }
    }
  }

  return std::make_tuple(max_val, min_val);
}

#if 0
int main() {
  // read the weight and bias of conv1
  std::string weight_dir= "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/model_weights/";
  std::string gold_input_fn = weight_dir + "tiny_en/gold_input.gaibin";
  std::string conv1_weight_fn = weight_dir + "tiny_en/encoder_conv1_weight.gaibin";
  std::string conv1_bias_fn = weight_dir + "tiny_en/encoder_conv1_bias.gaibin";
  std::string gold_conv1_fn = "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/gold_conv1.bin";
  std::map<std::string, int64_t> modelInfo {
    {conv1_weight_fn, 384 * 80 * 3},
    {conv1_bias_fn, 384},
  };

  const auto input = read_binary<float>(gold_input_fn, 1 * 80 * 3000);
  const auto conv1_bias = read_binary<float>(conv1_bias_fn, 384);
  const auto conv1_weight = read_binary<float>(conv1_weight_fn, 384 * 80 * 3);
  auto result = conv1d(input, conv1_weight, conv1_bias);

  /* std::cout << "conv1_bias = " << conv1_bias; */
  /* std::cout << "conv1_weight = " << conv1_weight; */
  const auto gold_conv1 = read_binary<float>(gold_conv1_fn, 384 * 300);
  float max_error = FLT_MIN;
  for (int i = 0; i < gold_conv1.size(); i++) {
    max_error = std::abs(gold_conv1[i] - result[i]) > max_error ? std::abs(gold_conv1[i] - result[i]) : max_error;
    /* std::cout << "(gold, result) = " << gold_conv1[i] << ", " << result[i] << "\n"; */
  }

  if (max_error < 1e05)
    std::cout << "(first conv1) PASS." << std::endl;
  else
    std::cout << "(first conv1) FAIL." << std::endl;


  return 0;
}
#endif
