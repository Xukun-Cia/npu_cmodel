#ifndef OPERATOR_CONV3D_PATCHED_H
#define OPERATOR_CONV3D_PATCHED_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conv3D Operator for Pre-Patched Input (ViT Patch Embedding)
//
// Input format: [num_patches, C_in, T, K_H, K_W] - already split into patches
// Weight format: [C_out, C_in, T, K_H, K_W]
// Bias format: [C_out]
// Output format: [num_patches, C_out]
//
// Operation: output = sum_t(input_t @ weight_t) + bias
//
// Precision:
// - FP16 input: can use FP16 or FP32 accumulator (controlled by use_fp16_accu)
// - BF16 input: always uses FP32 accumulator
//
// Parameters:
//   cycle_cnt:    [in/out] cycle counter
//   type:         NPU_TYPE_FP16 or NPU_TYPE_BF16
//   num_patches:  number of patches (M dimension)
//   C_in:         input channels
//   T:            temporal dimension (must be 2)
//   K_H, K_W:     kernel spatial dimensions
//   C_out:        output channels
//   input_ddr:    input tensor in DDR
//   weight_ddr:   weight tensor in DDR
//   bias_ddr:     bias vector in DDR (can be NULL)
//   rslt_ddr:     output tensor in DDR
//   use_fp16_accu: only valid for FP16, use FP16 accumulator if true (default false)
void user_operator_conv3d_patched(int* cycle_cnt, npu_datatype_t type,
                                   int num_patches,
                                   int C_in,
                                   int T,
                                   int K_H, int K_W,
                                   int C_out,
                                   void* input_ddr,
                                   void* weight_ddr,
                                   void* bias_ddr,
                                   void* rslt_ddr,
                                   bool use_fp16_accu);

#ifdef __cplusplus
}
#endif

#endif // OPERATOR_CONV3D_PATCHED_H
