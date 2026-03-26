#ifndef NPU_TPU_H
#define NPU_TPU_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// TPU Matrix Operations
// Calculates 16x16x16 Matrix Multiplication
// A: [16, 16], B: [16, 16] -> D: [16, 16]
// "mnk = 16x16x16" means A is MxK, B is KxN, D is MxN.
// Here M=16, N=16, K=16.
//
// Accumulator precision:
// - FP16 input: can use FP16 or FP32 accumulator (controlled by use_fp16_accu)
// - BF16 input: always uses FP32 accumulator
// - FP8 input: always uses FP32 accumulator
//
// D tensor type:
// - When use_fp16_accu=true (FP16 only): D is FP16
// - Otherwise: D is FP32
// Caller is responsible for prologue (convert bias/init to FP32) and 
// epilogue (convert FP32 result back to original type).

// D += A * B (Accumulate)
// - dst: FP32 tensor (or FP16 if use_fp16_accu=true for FP16 input)
// - src_a, src_b: input tensors in specified format (FP16/BF16/FP8)
// - format: input data format
// - use_fp16_accu: only valid for FP16, use FP16 accumulator if true
void tpu_matmul(npu_tensor_t* dst, const npu_tensor_t* src_a, const npu_tensor_t* src_b, 
                int* t, npu_datatype_t format, bool use_fp16_accu);

// D = A * B + C
// - dst: FP32 tensor (or FP16 if use_fp16_accu=true for FP16 input)
// - src_a, src_b: input tensors in specified format
// - src_c: same type as dst (FP32 or FP16)
// - format: input data format
// - use_fp16_accu: only valid for FP16, use FP16 accumulator if true
void tpu_matmul_add(npu_tensor_t* dst, const npu_tensor_t* src_a, const npu_tensor_t* src_b, 
                    const npu_tensor_t* src_c, int* t, npu_datatype_t format, bool use_fp16_accu);

#ifdef __cplusplus
}
#endif

#endif // NPU_TPU_H
