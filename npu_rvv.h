#ifndef NPU_RVV_H
#define NPU_RVV_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration
void rvv_configure(int lmul, int vlen_bits);

// Vector Arithmetic
// Performs element-wise operation on vectors.
// Length is determined by tensor size (simplified) or internal VL register (more accurate).
// For this C-model, we will operate on the number of elements in the destination tensor,
// clamped by the theoretical VLMAX = (VLEN/SEW) * LMUL.

void rvv_vadd(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void rvv_vsub(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void rvv_vfmul(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void rvv_vdiv(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

// Multiply-Accumulate: Dst = Dst + Src1 * Src2
void rvv_vfmacc(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

// Multiply-Accumulate with Scalar: Dst = Dst + scalar * Src
void rvv_vfmacc_vf(npu_tensor_t* dst, float scalar, const npu_tensor_t* src, int* t);

// Negative Multiply-Subtract-Accumulate: Dst = Dst - Src1 * Src2
void rvv_vfnmsac(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

// Negative Multiply-Subtract-Accumulate with Scalar: Dst = Dst - scalar * Src
void rvv_vfnmsac_vf(npu_tensor_t* dst, float scalar, const npu_tensor_t* src, int* t);

// Add with Scalar: Dst = Src + scalar
void rvv_vfadd_vf(npu_tensor_t* dst, const npu_tensor_t* src, float scalar, int* t);

// Multiply with Scalar: Dst = Src * scalar
void rvv_vfmul_vf(npu_tensor_t* dst, const npu_tensor_t* src, float scalar, int* t);

// Negate: Dst = -Src
void rvv_vfneg(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

// Absolute Value: Dst = |Src|
void rvv_vfabs(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

// Broadcast Scalar: Dst = scalar (all elements)
void rvv_vfmv_v_f(npu_tensor_t* dst, float scalar, int* t);

// Compare Greater Than (mask): returns mask
// For simplicity, we'll use a separate function that modifies dst based on comparison
void rvv_vmfgt(npu_tensor_t* dst, const npu_tensor_t* src1, float scalar, int* t);

// Compare Less or Equal (mask)
void rvv_vmfle(npu_tensor_t* dst, const npu_tensor_t* src1, float scalar, int* t);

// Merge: Dst = mask ? Src1 : Src2
void rvv_vmerge(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, const npu_tensor_t* mask, int* t);

// Reductions
// Vector Reduce Sum (produces scalar in dst[0])
void rvv_vredsum(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

// Max
void rvv_vfmax(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

// Load/Store (Vector load/store)
// Essentially similar to DMA but explicitly from vector unit perspective
void rvv_vle(npu_tensor_t* dst, const npu_tensor_t* src, int* t); // Vector Load
void rvv_vse(npu_tensor_t* dst, const npu_tensor_t* src, int* t); // Vector Store

// Shift Left Logical (Vector-Scalar, Integer)
// Dst = Src << shift
void rvv_vsll_vx(npu_tensor_t* dst, const npu_tensor_t* src, int shift, int* t);

// Integer Add (Vector-Scalar)
// Dst = Src + scalar
void rvv_vadd_vx(npu_tensor_t* dst, const npu_tensor_t* src, int scalar, int* t);

// Integer Subtract (Vector-Vector)
// Dst = Src1 - Src2
void rvv_vsub_vv(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

// Reinterpret Cast (Float <-> Uint32)
// This is a bit-wise copy, effectively a move between register types
void rvv_vreinterpret(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

// Population Count of Mask
// Returns number of set bits in the mask
int rvv_vcpop_m(const npu_tensor_t* mask, int* t);

// Move Integer Scalar to Vector
// Dst = scalar (all elements)
void rvv_vmv_v_x(npu_tensor_t* dst, int scalar, int* t);

// Integer Merge with Mask
// Dst = mask ? Src1 : Src2 (where Src2 is scalar)
// Note: Reference uses vmerge_vxm (vector-scalar-mask) where "Src2" is effectively the vector because of operand order in intrinsics?
// Reference: __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
// Wait, intrinsic is: vmerge_vxm(vector_false, scalar_true, mask) ?
// Checking standard: vmerge.vxm vd, vs2, rs1, v0
// if (v0.bit) vd = rs1 (scalar) else vd = vs2 (vector)
// The reference call: __riscv_vmerge_vxm_u32m2(vector_false, scalar_true, mask, vl)
// So if mask is 1, use scalar. If mask is 0, use vector.
// We will implement: rvv_vmerge_vxm(dst, src_vector_false, scalar_true, mask, t)
void rvv_vmerge_vxm(npu_tensor_t* dst, const npu_tensor_t* src_false, int scalar_true, const npu_tensor_t* mask, int* t);

// Vector Load/Store 32-bit elements (Explicit register load/store)
void rvv_vle32_v(npu_tensor_t* dst, const npu_tensor_t* src, int* t);
void rvv_vse32_v(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

// Vector Type Conversion (Cast) Instructions
// FP16 <-> FP32
void rvv_vfcvt_f32_fp16(npu_tensor_t* dst_f32, const npu_tensor_t* src_fp16, int* t);
void rvv_vfcvt_fp16_f32(npu_tensor_t* dst_fp16, const npu_tensor_t* src_f32, int* t);

// BF16 <-> FP32
void rvv_vfcvt_f32_bf16(npu_tensor_t* dst_f32, const npu_tensor_t* src_bf16, int* t);
void rvv_vfcvt_bf16_f32(npu_tensor_t* dst_bf16, const npu_tensor_t* src_f32, int* t);

#ifdef __cplusplus
}
#endif

#endif // NPU_RVV_H
