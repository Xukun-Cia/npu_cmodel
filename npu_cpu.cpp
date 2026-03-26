#include "npu_cpu.h"
#include <stdio.h>

// Helper to get scalar float
static float get_scalar_f32(const npu_tensor_t* t) {
    if (!t->data) return 0.0f;
    if (t->type == NPU_TYPE_FP32) return ((float*)t->data)[0];
    if (t->type == NPU_TYPE_INT32) return (float)((int32_t*)t->data)[0];
    // ... support other types if needed
    return 0.0f;
}

// Helper to set scalar float
static void set_scalar_f32(npu_tensor_t* t, float val) {
    if (!t->data) return;
    if (t->type == NPU_TYPE_FP32) ((float*)t->data)[0] = val;
    else if (t->type == NPU_TYPE_INT32) ((int32_t*)t->data)[0] = (int32_t)val;
}

void cpu_add(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    float v1 = get_scalar_f32(src1);
    float v2 = get_scalar_f32(src2);
    set_scalar_f32(dst, v1 + v2);
    
    if (t) *t += 1;
    g_npu_stats.inst_cpu++;
}

void cpu_sub(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    float v1 = get_scalar_f32(src1);
    float v2 = get_scalar_f32(src2);
    set_scalar_f32(dst, v1 - v2);
    
    if (t) *t += 1;
    g_npu_stats.inst_cpu++;
}

void cpu_mul(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    float v1 = get_scalar_f32(src1);
    float v2 = get_scalar_f32(src2);
    set_scalar_f32(dst, v1 * v2);
    
    if (t) *t += 1;
    g_npu_stats.inst_cpu++;
}

void cpu_div(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t) {
    float v1 = get_scalar_f32(src1);
    float v2 = get_scalar_f32(src2);
    set_scalar_f32(dst, (v2 != 0) ? v1 / v2 : 0.0f);
    
    if (t) *t += 1; // integer div might be more, but scalar fp div is usually 1-few on simplied cpu
    g_npu_stats.inst_cpu++;
}

