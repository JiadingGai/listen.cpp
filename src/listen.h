#ifndef LISTEN_H
#define LISTEN_H

#include <fstream>
#include <gperftools/profiler.h>

#define Entry1D(b,i) (*((b->entries)+i))
#define Entry2D(x,i,j)   (*((x->entries)+i*(x->dim1)+j))
#define Entry3D(d,i,j,k) (*((d->entries)+i*(d->dim1)*(d->dim2)+j*(d->dim2)+k))

typedef struct
{
  // Only support up to 3D tensor: good enough for whisper.
  // shape = (dim0, dim1, dim2)
  int dim0; //width;
  int dim1; //height;
  int dim2; //depth;
  float* entries;
} MATRIX_T;


MATRIX_T* Matrix_Allocate(int width, int height, int depth);

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

template<typename DataType>
MATRIX_T * read_binary_c(std::string Fn, int64_t numElements, const std::vector<int> &tensor_shape) {
  int64_t dataTypeSize = sizeof(DataType);
  int64_t BUFFERSIZE = numElements * dataTypeSize;
  char result_buffer[BUFFERSIZE];
  std::ifstream is(Fn, std::ios::in | std::ios::binary);
  assert(is && "input file not found!");
  is.read(result_buffer, BUFFERSIZE);
  std::vector<float> result;
  for (int i = 0; i < numElements; i++) {
    result.push_back(reinterpret_cast<DataType *>(result_buffer)[i]);
  }

  // convert result to MATRIX_T
  MATRIX_T* x = nullptr;
  if (tensor_shape.size() == 1) {
    x = Matrix_Allocate(1, 1, tensor_shape[0]);
  } else if (tensor_shape.size() == 2) {
    x = Matrix_Allocate(1, tensor_shape[0], tensor_shape[1]);
  } else if (tensor_shape.size() == 3) {
    x = Matrix_Allocate(tensor_shape[0], tensor_shape[1], tensor_shape[2]);
  } else {
    assert(false);
  }

  int count = 0;
  for (int i = 0; i < x->dim0; i++) {
    for (int j = 0; j < x->dim1; j++) {
      for (int k = 0; k < x->dim2; k++) {
        Entry3D(x, i, j, k) = result[count++];
      }
    }
  }

  return x;
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
void conv1d_c(MATRIX_T *out, MATRIX_T *in, MATRIX_T *w, MATRIX_T *bias, int stride);
std::vector<float> gelu(const std::vector<float> &input);
std::tuple<float, float, float> compare_matrix(MATRIX_T *x, MATRIX_T *y);
std::tuple<float, float> compute_range(MATRIX_T *x);


#endif
