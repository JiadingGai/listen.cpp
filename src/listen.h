#ifndef LISTEN_H
#define LISTEN_H

#include <fstream>
#include <gperftools/profiler.h>

#define Entry1D(b,i) (*((b->entries)+i))
#define Entry2D(x,i,j)   (*((x->entries)+i*(x->width)+j))
#define Entry3D(d,i,j,k) (*((d->entries)+k*(d->width)*(d->height)+i*(d->width)+j))

typedef struct
{
  int width;
  int height;
  int depth;
  float* entries;
} MATRIX_T;

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

std::vector<float> cross_correlation(const std::vector<float> &in, const std::vector<float> &w, int stride=1);
std::vector<float> add_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2);
std::vector<float> conv1d(const std::vector<float> &in, const std::vector<int> &in_shape, const std::vector<float> &w, const std::vector<int> &w_shape, const std::vector<float> &bias, int stride=1);
std::vector<float> gelu(const std::vector<float> &input);

#endif
