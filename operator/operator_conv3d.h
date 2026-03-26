#ifndef OPERATOR_CONV3D_H
#define OPERATOR_CONV3D_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conv3D Operator (ViT Patch Embedding)
// Implements Conv3D by decomposing into two Conv2D operations (via im2col->matmul) and summing
// For Qwen3-VL-4B ViT: T=2, Stride_T=2, Stride_H/W = Kernel_H/W (no overlap)
// 
// D = im2col(input_t0) * weight_t0 + im2col(input_t1) * weight_t1 + bias
//
// Parameters:
//   cycle_cnt  - Accumulated cycle count
//   type       - Data type: NPU_TYPE_FP16 or NPU_TYPE_BF16
//   H, W       - Input spatial dimensions (e.g., 448x448)
//   C_in       - Input channels (e.g., 3 for RGB)
//   T          - Temporal dimension (e.g., 2)
//   K_H, K_W   - Kernel spatial dimensions (e.g., 16x16)
//   C_out      - Output channels (e.g., 1024)
//   input_ddr  - Input data [T, H, W, C_in] in DDR
//   weight_ddr - Weight data [C_out, C_in, T, K_H, K_W] in DDR
//   bias_ddr   - Bias data [C_out] in DDR (can be NULL)
//   rslt_ddr   - Output data [num_patches, C_out] in DDR
//
// Note: Assumes Stride_H = K_H, Stride_W = K_W, Stride_T = T (no overlap)
//       num_patches = (H / K_H) * (W / K_W)

void user_operator_conv3d(int* cycle_cnt, npu_datatype_t type,
                          int H, int W, int C_in,
                          int T,
                          int K_H, int K_W,
                          int C_out,
                          void* input_ddr,
                          void* weight_ddr,
                          void* bias_ddr,
                          void* rslt_ddr);

#ifdef __cplusplus
}
#endif

#endif // OPERATOR_CONV3D_H

