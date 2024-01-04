#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <cfloat>

template<typename DataType>
std::vector<float> read_binary(std::string weightFn, int64_t numElements) {
  int64_t dataTypeSize = sizeof(DataType);
  int64_t BUFFERSIZE = numElements * dataTypeSize;
  char result_buffer[BUFFERSIZE];
  std::ifstream is(weightFn, std::ios::in | std::ios::binary);
  assert(is && "input file not found!");
  is.read(result_buffer, BUFFERSIZE);
  std::vector<float> result;
  for (int i = 0; i < numElements; i++) {
    result.push_back(reinterpret_cast<float *>(result_buffer)[i]);
  }

  return result;
}

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii) os << ", " << *ii;
  os << "]\n";
  return os;
}

std::vector<float> cross_correlation(const std::vector<float> &in, const std::vector<float> &w) {
  std::vector<float> out;
  assert(in.size() == 3000);
  assert(w.size() == 3);

  for (int i = 0; i < in.size(); i++) {
    auto tmp = in[i] * w[1];
    if (i - 1 >= 0)
      tmp += in[i-1] * w[0];

    if (i + 1 < in.size())
      tmp += in[i+1] * w[2];

    out.push_back(tmp);
  }

  return out;
}

std::vector<float> add_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    std::vector<float> result;

    // Check if vectors are of the same size
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors must be of the same size for addition." << std::endl;
        return result; // Return an empty vector if sizes are different
    }

    // Add corresponding elements from both vectors
    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] + vec2[i]);
    }

    return result;
}

std::vector<float> conv1d(const std::vector<float> &in, const std::vector<float> &w, const std::vector<float> &bias) {
  // https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
  // in : [1, 80, 3000]
  // w: [384,80, 3], ks=(3,1), stride=(1,), padding=1
  // b: [384]
  // out: [1, 384, 3000]

  //out shape = [1, 384, 3000]
  std::vector<float> out(1*384*3000);
  for (int b = 0; b < 1; b++) { //batch dim
    for (int oc = 0; oc < 384; oc++) { // out channel dim
      std::vector<float> out_tmp(3000, 0);
      for (int k = 0; k < 80; k++) {
        std::vector<float> w_piece(w.begin() + oc * 80 * 3 + k * 3, w.begin() + oc * 80 * 3 + k * 3 + 3);
        std::vector<float> input_piece(in.begin() + b * 80 * 3000 + k * 3000, in.begin() + b * 80 * 3000 + k * 3000 + 3000);
        out_tmp = add_vectors(cross_correlation(input_piece, w_piece), out_tmp);
      }
      // add bias
      for (auto &x : out_tmp) {
        x += bias[oc];
      }
      //save out_tmp back to out.
      std::copy(out_tmp.begin(), out_tmp.end(), out.begin() + b * 384 * 3000 + oc * 3000);
    }
  }

  return out;
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
