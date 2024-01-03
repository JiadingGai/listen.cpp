#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>

template<typename DataType>
std::vector<float> read_weights(std::string weightFn, int64_t numElements) {
  int64_t dataTypeSize = sizeof(DataType);
  int64_t BUFFERSIZE = numElements * dataTypeSize;
  char bias_buffer[BUFFERSIZE];
  std::ifstream is(weightFn, std::ios::in | std::ios::binary);
  is.read(bias_buffer, BUFFERSIZE);
  std::vector<float> bias;
  for (int i = 0; i < numElements; i++) {
    bias.push_back(reinterpret_cast<float *>(bias_buffer)[i]);
  }

  return bias;
}

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii) os << ", " << *ii;
  os << "]\n";
  return os;
}

int main() {

  // read the weight and bias of conv1
  std::string weight_dir= "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/model_weights/";
  std::string conv1_weight_fn = weight_dir + "tiny_en/encoder_conv1_weight.gaibin";
  std::string conv1_bias_fn = weight_dir + "tiny_en/encoder_conv1_bias.gaibin";
  std::map<std::string, int64_t> modelInfo {
    {conv1_weight_fn, 384 * 80 * 3},
    {conv1_bias_fn, 384},
  };

  const auto conv1_bias = read_weights<float>(conv1_bias_fn, 384);
  const auto conv1_weight = read_weights<float>(conv1_weight_fn, 384 * 80 * 3);

  std::cout << "conv1_bias = " << conv1_bias;
  /* std::cout << "conv1_weight = " << conv1_weight; */

  return 0;
}
