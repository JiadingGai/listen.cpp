#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_universal.h"

// This example shows how to invoke a cutlass unittest in a standalone program.
// Reference: https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_universal_bf16t_s8n_f32t_mixed_input_tensor_op_f32_sm80.cu 

// nvcc -o __gai gai_demo.cu -I cutlass/include -I cutlass/tools/util/include -I cutlass/test/unit/gemm/device --expt-relaxed-constexpr -I build/_deps/googletest-src/googletest/include -L build/lib -lgtest -arch=compute_80 -code=sm_80

void test_f16t_s8n()
{
  // f16t_s8n
  //gemm_universal_f16t_s8n_f16t_mixed_input_tensor_op_f32_sm80.cu
  // D = alpha x AB + beta x C
  using ElementA = cutlass::half_t;
  using ElementB = int8_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,  // Stages
    8,  // AlignmentA
    16, // AlignmentB
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone
  >;
  
  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

void test_s4t_s8n_s32t()
{
  using ElementA = cutlass::int4b_t;
  using ElementB = int8_t;
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,  // Stages
    8,  // AlignmentA
    16, // AlignmentB 
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

int main()
{
  test_s4t_s8n_s32t();
  test_f16t_s8n();

#if 0
  test::gemm::device::TestAllGemmUniversal<Gemm>();
  test::gemm::device::TestbedUniversal<Gemm, /*Relu*/false> testbed;
  double alpha = 1.0, beta = 2.0;
  int batch_count = 5;
  int m = 128, n = 128, k = 128;
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
  bool passed = testbed.run(
      mode,
      problem_size,
      batch_count,
      cutlass::from_real<ElementCompute>(alpha),
      cutlass::from_real<ElementCompute>(beta)
  );
#endif

  return 0;
}
