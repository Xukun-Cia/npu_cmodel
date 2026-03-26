#include "npu_tpu.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// Helper to access matrix element (row, col) - FP16 version
static float get_val_fp16(const npu_tensor_t* t, int row, int col) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    return fp16_to_fp32(*(uint16_t*)ptr);
}

static void set_val_fp16(npu_tensor_t* t, int row, int col, float v) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    *(uint16_t*)ptr = fp32_to_fp16(v);
}

// Helper to access matrix element (row, col) - BF16 version
static float get_val_bf16(const npu_tensor_t* t, int row, int col) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    return bf16_to_fp32(*(uint16_t*)ptr);
}

// Helper to access matrix element (row, col) - FP8 version
static float get_val_fp8(const npu_tensor_t* t, int row, int col) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    return fp8_e4m3_to_fp32(*(uint8_t*)ptr);
}

// Helper to access matrix element (row, col) - FP32 version
static float get_val_fp32(const npu_tensor_t* t, int row, int col) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    return *(float*)ptr;
}

static void set_val_fp32(npu_tensor_t* t, int row, int col, float v) {
    size_t offset = row * t->nb[1] + col * t->nb[0];
    char* ptr = (char*)t->data + offset;
    *(float*)ptr = v;
}

// D += A * B
// D is FP32 (or FP16 if use_fp16_accu=true for FP16 input)
// A, B are in specified format
void tpu_matmul(npu_tensor_t* dst, const npu_tensor_t* src_a, const npu_tensor_t* src_b, 
                int* t, npu_datatype_t format, bool use_fp16_accu) {
    
    if (format != NPU_TYPE_FP16 && format != NPU_TYPE_BF16 && format != NPU_TYPE_FP8) {
        fprintf(stderr, "ERROR: tpu_matmul only supports FP16, BF16, and FP8\n");
        return;
    }
    
    // For BF16 and FP8, use_fp16_accu must be false
    if (format != NPU_TYPE_FP16 && use_fp16_accu) {
        assert(false && "use_fp16_accu=true is only valid for FP16 format");
        return;
    }
    
    if (format == NPU_TYPE_FP16 && use_fp16_accu) {
        // FP16 input with FP16 accumulator
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = get_val_fp16(dst, m, n);
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp16(src_a, m, k) * get_val_fp16(src_b, k, n);
                }
                set_val_fp16(dst, m, n, sum);
            }
        }
    } else if (format == NPU_TYPE_FP16) {
        // FP16 input with FP32 accumulator
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = get_val_fp32(dst, m, n);
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp16(src_a, m, k) * get_val_fp16(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum);
            }
        }
    } else if (format == NPU_TYPE_BF16) {
        // BF16 input with FP32 accumulator (mandatory)
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = get_val_fp32(dst, m, n);
                for (int k = 0; k < 16; k++) {
                    sum += get_val_bf16(src_a, m, k) * get_val_bf16(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum);
            }
        }
    } else { // NPU_TYPE_FP8
        // FP8 input with FP32 accumulator (mandatory)
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = get_val_fp32(dst, m, n);
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp8(src_a, m, k) * get_val_fp8(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum);
            }
        }
    }
    
    // Cycle count
    if (t) {
        if (format == NPU_TYPE_FP8) {
            *t += 16;
        } else {
            *t += 64;
        }
    }
    
    g_npu_stats.inst_tpu++;
}

// D = A * B + C
// D and C are FP32 (or FP16 if use_fp16_accu=true for FP16 input)
// A, B are in specified format
void tpu_matmul_add(npu_tensor_t* dst, const npu_tensor_t* src_a, const npu_tensor_t* src_b, 
                    const npu_tensor_t* src_c, int* t, npu_datatype_t format, bool use_fp16_accu) {
    
    if (format != NPU_TYPE_FP16 && format != NPU_TYPE_BF16 && format != NPU_TYPE_FP8) {
        fprintf(stderr, "ERROR: tpu_matmul_add only supports FP16, BF16, and FP8\n");
        return;
    }
    
    // For BF16 and FP8, use_fp16_accu must be false
    if (format != NPU_TYPE_FP16 && use_fp16_accu) {
        assert(false && "use_fp16_accu=true is only valid for FP16 format");
        return;
    }
    
    if (format == NPU_TYPE_FP16 && use_fp16_accu) {
        // FP16 input with FP16 accumulator
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp16(src_a, m, k) * get_val_fp16(src_b, k, n);
                }
                set_val_fp16(dst, m, n, sum + get_val_fp16(src_c, m, n));
            }
        }
    } else if (format == NPU_TYPE_FP16) {
        // FP16 input with FP32 accumulator
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp16(src_a, m, k) * get_val_fp16(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum + get_val_fp32(src_c, m, n));
            }
        }
    } else if (format == NPU_TYPE_BF16) {
        // BF16 input with FP32 accumulator (mandatory)
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += get_val_bf16(src_a, m, k) * get_val_bf16(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum + get_val_fp32(src_c, m, n));
            }
        }
    } else { // NPU_TYPE_FP8
        // FP8 input with FP32 accumulator (mandatory)
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += get_val_fp8(src_a, m, k) * get_val_fp8(src_b, k, n);
                }
                set_val_fp32(dst, m, n, sum + get_val_fp32(src_c, m, n));
            }
        }
    }
    
    if (t) {
        if (format == NPU_TYPE_FP8) {
            *t += 16;
        } else {
            *t += 64;
        }
    }
    
    g_npu_stats.inst_tpu++;
}
