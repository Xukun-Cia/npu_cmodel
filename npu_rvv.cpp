#include "npu_rvv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int g_lmul = 1;
static int g_vlen_bits = 128; // Default VLEN 128 bits

void rvv_configure(int lmul, int vlen_bits) {
    if (lmul != 1 && lmul != 2 && lmul != 4 && lmul != 8) {
        fprintf(stderr, "Warning: Invalid LMUL %d, defaulting to 1\n", lmul);
        g_lmul = 1;
    } else {
        g_lmul = lmul;
    }
    
    if (vlen_bits > 0) g_vlen_bits = vlen_bits;
}

// Helper to get VLMAX (max elements per vector instruction)
static int get_vlmax(npu_datatype_t type) {
    int esize = npu_type_size(type); // Element size in bytes
    int vlen_bytes = g_vlen_bits / 8;
    return (vlen_bytes / esize) * g_lmul;
}

static float get_f32(const npu_tensor_t* t, int i) {
    // Flat index access
    // Simplified: assume contiguous data for RVV ops
    if (!t->data) return 0.0f;
    float* f_data = (float*)t->data; // Assume float storage
    return f_data[i];
}

static void set_f32(npu_tensor_t* t, int i, float val) {
    if (!t->data) return;
    float* f_data = (float*)t->data;
    f_data[i] = val;
}

// Helpers for Integer Ops
static uint32_t get_u32(const npu_tensor_t* t, int i) {
    if (!t->data) return 0;
    uint32_t* data = (uint32_t*)t->data;
    return data[i];
}

static void set_u32(npu_tensor_t* t, int i, uint32_t val) {
    if (!t->data) return;
    uint32_t* data = (uint32_t*)t->data;
    data[i] = val;
}

// Generic Element-wise Operation Template
typedef float (*binary_op_t)(float, float);

static void rvv_binary_op(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t, binary_op_t op) {
    int64_t n = npu_tensor_ne(dst);
    
    for (int64_t i = 0; i < n; i++) {
        float v1 = get_f32(src1, i);
        float v2 = get_f32(src2, i);
        set_f32(dst, i, op(v1, v2));
    }
    
    // Calculate cycles
    // How many vector instructions needed?
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts; // Assume 1 cycle throughput per instruction
    
    g_npu_stats.inst_rvv += num_insts;
}

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) { return (b != 0) ? a / b : 0.0f; }
static float op_max(float a, float b) { return (a > b) ? a : b; }

void rvv_vadd(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    rvv_binary_op(dst, src1, src2, t, op_add);
}

void rvv_vsub(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    rvv_binary_op(dst, src1, src2, t, op_sub);
}

void rvv_vfmul(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    rvv_binary_op(dst, src1, src2, t, op_mul);
}

void rvv_vdiv(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    rvv_binary_op(dst, src1, src2, t, op_div);
}

void rvv_vfmax(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    rvv_binary_op(dst, src1, src2, t, op_max);
}

void rvv_vfmacc(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float d = get_f32(dst, i);
        float v1 = get_f32(src1, i);
        float v2 = get_f32(src2, i);
        set_f32(dst, i, d + (v1 * v2));
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Multiply-Accumulate with Scalar: Dst = Dst + scalar * Src
void rvv_vfmacc_vf(npu_tensor_t* dst, float scalar, const npu_tensor_t* src, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float d = get_f32(dst, i);
        float v = get_f32(src, i);
        set_f32(dst, i, d + (scalar * v));
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vredsum(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    int64_t n = npu_tensor_ne(src);
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        sum += get_f32(src, i);
    }
    
    // Result is scalar in dst[0]
    set_f32(dst, 0, sum);
    
    // Reduction takes logarithmic steps usually, or linear in simplifed model
    int vlmax = get_vlmax(src->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    int cycles = num_insts * 2; // Heuristic
    
    if (t) *t += cycles;
    g_npu_stats.inst_rvv += num_insts; 
}

void rvv_vfnmsac(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float d = get_f32(dst, i);
        float v1 = get_f32(src1, i);
        float v2 = get_f32(src2, i);
        set_f32(dst, i, d - (v1 * v2));
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Negative Multiply-Subtract-Accumulate with Scalar: Dst = Dst - scalar * Src
void rvv_vfnmsac_vf(npu_tensor_t* dst, float scalar, const npu_tensor_t* src, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float d = get_f32(dst, i);
        float v = get_f32(src, i);
        set_f32(dst, i, d - (scalar * v));
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Add with Scalar: Dst = Src + scalar
void rvv_vfadd_vf(npu_tensor_t* dst, const npu_tensor_t* src, float scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src, i);
        set_f32(dst, i, v + scalar);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Multiply with Scalar: Dst = Src * scalar
void rvv_vfmul_vf(npu_tensor_t* dst, const npu_tensor_t* src, float scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src, i);
        set_f32(dst, i, v * scalar);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vfneg(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src, i);
        set_f32(dst, i, -v);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vfabs(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src, i);
        set_f32(dst, i, (v < 0) ? -v : v);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vfmv_v_f(npu_tensor_t* dst, float scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        set_f32(dst, i, scalar);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Compare Greater Than: dst[i] = (src1[i] > scalar) ? 1.0f : 0.0f
void rvv_vmfgt(npu_tensor_t* dst, const npu_tensor_t* src1, float scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src1, i);
        set_f32(dst, i, (v > scalar) ? 1.0f : 0.0f);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Compare Less or Equal: dst[i] = (src1[i] <= scalar) ? 1.0f : 0.0f
void rvv_vmfle(npu_tensor_t* dst, const npu_tensor_t* src1, float scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = get_f32(src1, i);
        set_f32(dst, i, (v <= scalar) ? 1.0f : 0.0f);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Merge: dst[i] = (mask[i] != 0) ? src1[i] : src2[i]
void rvv_vmerge(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, const npu_tensor_t* mask, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float m = get_f32(mask, i);
        float v1 = get_f32(src1, i);
        float v2 = get_f32(src2, i);
        set_f32(dst, i, (m != 0.0f) ? v1 : v2);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vle(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    // Vector Load
    // Treat as copy
    int64_t n = npu_tensor_ne(dst);
    size_t bytes = n * npu_type_size(dst->type);
    if (dst->data && src->data) {
        // If src is in SRAM/DRAM and dst is "Register" (also SRAM in this model)
        memcpy(dst->data, src->data, bytes);
    }
    
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vse(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    // Vector Store
    int64_t n = npu_tensor_ne(src);
    size_t bytes = n * npu_type_size(src->type);
    if (dst->data && src->data) {
        memcpy(dst->data, src->data, bytes);
    }
    
    int vlmax = get_vlmax(src->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vsll_vx(npu_tensor_t* dst, const npu_tensor_t* src, int shift, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        uint32_t v = get_u32(src, i);
        set_u32(dst, i, v << shift);
    }
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vadd_vx(npu_tensor_t* dst, const npu_tensor_t* src, int scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        uint32_t v = get_u32(src, i);
        set_u32(dst, i, v + scalar);
    }
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vsub_vv(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        uint32_t v1 = get_u32(src1, i);
        uint32_t v2 = get_u32(src2, i);
        set_u32(dst, i, v1 - v2);
    }
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vreinterpret(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    // Bit copy
    int64_t n = npu_tensor_ne(dst);
    size_t bytes = n * 4; // Assuming 32-bit elements
    if (dst->data && src->data) {
        memcpy(dst->data, src->data, bytes);
    }
    // Can be considered 0 cycles in some architectures (rename) or move
    // We'll count as move
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

int rvv_vcpop_m(const npu_tensor_t* mask, int* t) {
    int64_t n = npu_tensor_ne(mask);
    int count = 0;
    for (int64_t i = 0; i < n; i++) {
        float m = get_f32(mask, i); // Mask is stored as float 0.0 or 1.0 in this model
        if (m != 0.0f) count++;
    }
    // Reduction cost
    int vlmax = get_vlmax(mask->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
    return count;
}

void rvv_vmv_v_x(npu_tensor_t* dst, int scalar, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        set_u32(dst, i, (uint32_t)scalar);
    }
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

void rvv_vmerge_vxm(npu_tensor_t* dst, const npu_tensor_t* src_false, int scalar_true, const npu_tensor_t* mask, int* t) {
    int64_t n = npu_tensor_ne(dst);
    for (int64_t i = 0; i < n; i++) {
        float m = get_f32(mask, i);
        // If mask is 1 (true), use scalar. Else use src_false.
        if (m != 0.0f) {
             set_u32(dst, i, (uint32_t)scalar_true);
        } else {
             set_u32(dst, i, get_u32(src_false, i));
        }
    }
    int vlmax = get_vlmax(dst->type);
    int num_insts = (n + vlmax - 1) / vlmax;
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// Explicit Load/Store (alias to vle/vse)
void rvv_vle32_v(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    rvv_vle(dst, src, t);
}

void rvv_vse32_v(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    rvv_vse(dst, src, t);
}

// Vector Type Conversion (Cast) Instructions
// FP16 -> FP32
void rvv_vfcvt_f32_fp16(npu_tensor_t* dst_f32, const npu_tensor_t* src_fp16, int* t) {
    int64_t n = npu_tensor_ne(src_fp16);
    uint16_t* src_data = (uint16_t*)src_fp16->data;
    float* dst_data = (float*)dst_f32->data;
    
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = fp16_to_fp32(src_data[i]);
    }
    
    int vlmax = get_vlmax(NPU_TYPE_FP16);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// FP32 -> FP16
void rvv_vfcvt_fp16_f32(npu_tensor_t* dst_fp16, const npu_tensor_t* src_f32, int* t) {
    int64_t n = npu_tensor_ne(src_f32);
    float* src_data = (float*)src_f32->data;
    uint16_t* dst_data = (uint16_t*)dst_fp16->data;
    
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = fp32_to_fp16(src_data[i]);
    }
    
    int vlmax = get_vlmax(NPU_TYPE_FP32);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// BF16 -> FP32
void rvv_vfcvt_f32_bf16(npu_tensor_t* dst_f32, const npu_tensor_t* src_bf16, int* t) {
    int64_t n = npu_tensor_ne(src_bf16);
    uint16_t* src_data = (uint16_t*)src_bf16->data;
    float* dst_data = (float*)dst_f32->data;
    
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = bf16_to_fp32(src_data[i]);
    }
    
    int vlmax = get_vlmax(NPU_TYPE_BF16);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}

// FP32 -> BF16
void rvv_vfcvt_bf16_f32(npu_tensor_t* dst_bf16, const npu_tensor_t* src_f32, int* t) {
    int64_t n = npu_tensor_ne(src_f32);
    float* src_data = (float*)src_f32->data;
    uint16_t* dst_data = (uint16_t*)dst_bf16->data;
    
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = fp32_to_bf16(src_data[i]);
    }
    
    int vlmax = get_vlmax(NPU_TYPE_FP32);
    int num_insts = (n + vlmax - 1) / vlmax;
    
    if (t) *t += num_insts;
    g_npu_stats.inst_rvv += num_insts;
}
