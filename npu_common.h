#ifndef NPU_COMMON_H
#define NPU_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hardware Configuration
#define NPU_NUM_CLUSTERS 4
#define NPU_TPU_ARRAYS_PER_CLUSTER 8
#define NPU_TPU_TOTAL_ARRAYS (NPU_NUM_CLUSTERS * NPU_TPU_ARRAYS_PER_CLUSTER)
#define NPU_TPU_ARRAY_WIDTH 16
#define NPU_TPU_ARRAY_HEIGHT 16
#define NPU_TPU_MAC_COUNT (NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT)
#define NPU_TPU_TOTAL_MAC_COUNT (NPU_TPU_TOTAL_ARRAYS * NPU_TPU_MAC_COUNT)
#define NPU_RVV_VLEN_DLEN 512
#define NPU_DMA_BANDWIDTH_BYTES 128 // 1024 bits = 128 bytes per cycle

// MAX Dimensions for tensor
#define MAX_DIMS 4

// Data Types
typedef enum {
    NPU_TYPE_FP32,
    NPU_TYPE_FP16,
    NPU_TYPE_BF16,
    NPU_TYPE_FP8,
    NPU_TYPE_INT8,
    NPU_TYPE_INT32
} npu_datatype_t;

// Simplified Tensor Structure (based on ggml_tensor)
typedef struct {
    npu_datatype_t type;
    int ndim;
    int64_t ne[MAX_DIMS];   // Number of elements in each dimension
    size_t  nb[MAX_DIMS];   // Stride in bytes (optional, but good for correctness)
    void*   data;           // Pointer to data (in SRAM/DRAM)
} npu_tensor_t;

// Global Statistics Structure
typedef struct {
    // Total cycles tracked via function arguments now
    int64_t total_inst;
    
    int64_t inst_cpu;
    int64_t inst_rvv;
    int64_t inst_tpu;
    int64_t inst_dma;
    int64_t inst_sram;
} npu_stats_t;

// Global instance of stats (defined in npu_top.cpp)
extern npu_stats_t g_npu_stats;

// Helper to get type size
static inline int npu_type_size(npu_datatype_t type) {
    switch (type) {
        case NPU_TYPE_FP32: return 4;
        case NPU_TYPE_FP16: return 2;
        case NPU_TYPE_BF16: return 2;
        case NPU_TYPE_FP8:  return 1;
        case NPU_TYPE_INT8: return 1;
        case NPU_TYPE_INT32: return 4;
        default: return 1;
    }
}

// Helper to get number of elements
static inline int64_t npu_tensor_ne(const npu_tensor_t* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) {
        n *= t->ne[i];
    }
    return n;
}

// Helper to create a tensor wrapper around data
static inline npu_tensor_t create_tensor(void* data, int ndim, int64_t* ne, npu_datatype_t type) {
    npu_tensor_t t;
    t.data = data;
    t.ndim = ndim;
    t.type = type;
    for (int i=0; i<ndim; i++) t.ne[i] = ne[i];
    
    // Calculate strides (standard dense)
    size_t esize = npu_type_size(type);
    t.nb[0] = esize;
    for (int i=1; i<ndim; i++) {
        t.nb[i] = t.nb[i-1] * t.ne[i-1];
    }
    return t;
}

// Helper to reset stats
static inline void stats_reset() {
    g_npu_stats.total_inst = 0;
    g_npu_stats.inst_cpu = 0;
    g_npu_stats.inst_rvv = 0;
    g_npu_stats.inst_tpu = 0;
    g_npu_stats.inst_dma = 0;
    g_npu_stats.inst_sram = 0;
}

// Helper to print stats
static inline void stats_print(int cycles) {
    int64_t total_inst = g_npu_stats.inst_cpu + g_npu_stats.inst_rvv + g_npu_stats.inst_tpu + g_npu_stats.inst_dma + g_npu_stats.inst_sram;

    printf(" Simulation Statistics:\n");
    printf(" Total  Cycles:  %d\n", cycles);
    printf(" Total  Instrs:  %ld\n", total_inst);
    printf(" - CPU  Instrs:  %ld\n", g_npu_stats.inst_cpu);
    printf(" - RVV  Instrs:  %ld\n", g_npu_stats.inst_rvv);
    printf(" - TPU  Instrs:  %ld\n", g_npu_stats.inst_tpu);
    printf(" - DMA  Instrs:  %ld\n", g_npu_stats.inst_dma);
    printf(" - SRAM Instrs:  %ld\n", g_npu_stats.inst_sram);
    printf("\n==============================================================\n");
}

// FP16 Format Conversion Helpers (IEEE 754 half precision)
// FP16: 16 bits, 1 sign bit, 5 exponent bits, 10 mantissa bits
// Bias: 15
// FP16 Format Conversion Helpers (using direct bit operations)
// FP16: 16 bits, 1 sign bit, 5 exponent bits, 10 mantissa bits
// Format: S EEEEE MMMMMMMMMM
static inline uint16_t fp32_to_fp16(float f) {
    union { uint32_t i; float f; } u;
    u.f = f;
    uint32_t bits = u.i;
    
    // Extract FP32 components
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;
    
    // Handle special FP32 values
    if (exp32 == 0xFF) {
        // Infinity or NaN
        if (mant32 == 0) {
            // Infinity: FP16 exp=31, mant=0
            return (sign << 15) | (31 << 10);
        } else {
            // NaN: FP16 exp=31, mant!=0
            return (sign << 15) | (31 << 10) | 1;
        }
    }
    
    // Convert exponent
    int32_t exp = (int32_t)exp32 - 127;
    int32_t fp16_exp = exp + 15;
    
    uint16_t fp16_sign = (uint16_t)sign;
    uint16_t fp16_exp_bits;
    uint16_t fp16_mant;
    
    if (fp16_exp > 31) {
        // Overflow: convert to infinity
        fp16_exp_bits = 31;
        fp16_mant = 0;
    } else if (fp16_exp < 0) {
        // Underflow: convert to zero or subnormal
        fp16_exp_bits = 0;
        if (exp < -14) {
            // Too small, underflow to zero
            fp16_mant = 0;
        } else {
            // Subnormal: keep mantissa
            fp16_mant = (uint16_t)(mant32 >> 13);
        }
    } else {
        // Normal number
        fp16_exp_bits = (uint16_t)fp16_exp;
        fp16_mant = (uint16_t)(mant32 >> 13);
    }
    
    return (fp16_sign << 15) | (fp16_exp_bits << 10) | fp16_mant;
}

static inline float fp16_to_fp32(uint16_t val) {
    // Parse FP16 directly using bit operations
    uint16_t sign = (val >> 15) & 0x1;
    uint16_t exponent = (val >> 10) & 0x1F;
    uint16_t mantissa = val & 0x3FF;
    
    // Handle zero
    if (exponent == 0 && mantissa == 0) {
        union { uint32_t i; float f; } u;
        u.i = (sign << 31);
        return u.f;
    }
    
    // Handle infinity/NaN (exp=31)
    if (exponent == 31) {
        union { uint32_t i; float f; } u;
        if (mantissa == 0) {
            // Infinity
            u.i = (sign << 31) | 0x7F800000;
        } else {
            // NaN
            u.i = (sign << 31) | 0x7FC00000;
        }
        return u.f;
    }
    
    // Handle subnormal (exp=0, mant!=0)
    if (exponent == 0) {
        // Subnormal: (-1)^S * 2^(-14) * (M/1024)
        float sign_val = sign ? -1.0f : 1.0f;
        float result = ((float)mantissa) / 1024.0f;
        result *= 0.00006103515625f; // 2^(-14)
        union { uint32_t i; float f; } u;
        u.f = sign_val * result;
        return u.f;
    }
    
    // Normal number: FP32 = sign | (exp+127-15)<<23 | mant<<13
    uint32_t fp32_exp = (uint32_t)(exponent + 127 - 15);
    uint32_t fp32_mantissa = ((uint32_t)mantissa) << 13;
    
    union { uint32_t i; float f; } u;
    u.i = (sign << 31) | (fp32_exp << 23) | fp32_mantissa;
    return u.f;
}

// BF16 Format Conversion Helpers (using direct bit operations)
// BF16: 16 bits, 1 sign bit, 8 exponent bits, 7 mantissa bits
// Format: S EEEEEEEE MMMMMMM
// Same exponent range as FP32, but lower mantissa precision
static inline uint16_t fp32_to_bf16(float val) {
    union { uint32_t i; float f; } u;
    u.f = val;
    uint32_t bits = u.i;
    
    // BF16 is simply the top 16 bits of FP32
    // sign (1 bit) + exponent (8 bits) + top 7 bits of mantissa
    uint16_t sign = (bits >> 31) & 0x1;
    uint16_t exponent = (bits >> 23) & 0xFF;
    uint16_t mantissa = (bits >> 16) & 0x7F;
    
    return (sign << 15) | (exponent << 7) | mantissa;
}

static inline float bf16_to_fp32(uint16_t val) {
    // Parse BF16 directly using bit operations
    uint16_t sign = (val >> 15) & 0x1;
    uint16_t exponent = (val >> 7) & 0xFF;
    uint16_t mantissa = val & 0x7F;
    
    // BF16 to FP32: sign | exp | mant | zeros (lower 16 bits are zero)
    union { uint32_t i; float f; } u;
    u.i = ((uint32_t)sign << 31) |
          ((uint32_t)exponent << 23) |
          ((uint32_t)mantissa << 16);
    return u.f;
}

// FP8 (E4M3) Format Conversion Helpers (using direct bit operations)
// Format: S EEEE MMM (1 sign bit, 4 exponent bits, 3 mantissa bits)
// Bias: 7
static inline uint8_t fp32_to_fp8_e4m3(float val) {
    union { uint32_t i; float f; } u;
    u.f = val;
    uint32_t bits = u.i;
    
    // Extract FP32 components
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;
    
    // Handle special FP32 values
    if (exp32 == 0xFF) {
        // Infinity or NaN
        if (mant32 == 0) {
            // Infinity: FP8 exp=15, mant=0
            return (sign << 7) | (15 << 3);
        } else {
            // NaN: FP8 exp=15, mant!=0
            return (sign << 7) | (15 << 3) | 1;
        }
    }
    
    // Convert exponent: FP32 bias=127, FP8 bias=7
    int32_t exp = (int32_t)exp32 - 127;
    int32_t fp8_exp = exp + 7;
    
    uint8_t fp8_sign = (uint8_t)sign;
    uint8_t fp8_exp_bits;
    uint8_t fp8_mant;
    
    if (fp8_exp > 15) {
        // Overflow: convert to infinity
        fp8_exp_bits = 15;
        fp8_mant = 0;
    } else if (fp8_exp < 0) {
        // Underflow: convert to zero or subnormal
        fp8_exp_bits = 0;
        if (exp < -7) {
            // Too small, underflow to zero
            fp8_mant = 0;
        } else {
            // Subnormal: keep top 3 bits of mantissa
            fp8_mant = (uint8_t)(mant32 >> 20);
        }
    } else {
        // Normal number
        fp8_exp_bits = (uint8_t)fp8_exp;
        fp8_mant = (uint8_t)(mant32 >> 20); // Take top 3 bits of mantissa
    }
    
    return (fp8_sign << 7) | (fp8_exp_bits << 3) | fp8_mant;
}

static inline float fp8_e4m3_to_fp32(uint8_t val) {
    // Parse FP8 directly using bit operations to avoid bitfield endianness issues
    uint8_t sign = (val >> 7) & 0x1;
    uint8_t exponent = (val >> 3) & 0xF;
    uint8_t mantissa = val & 0x7;
    
    // Handle zero
    if (exponent == 0 && mantissa == 0) {
        union { uint32_t i; float f; } u;
        u.i = (sign << 31);
        return u.f;
    }
    
    // Handle infinity/NaN (exp=15)
    if (exponent == 15) {
        union { uint32_t i; float f; } u;
        if (mantissa == 0) {
            // Infinity
            u.i = (sign << 31) | 0x7F800000;
        } else {
            // NaN
            u.i = (sign << 31) | 0x7FC00000;
        }
        return u.f;
    }
    
    // Normal number: value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
    // FP32 format: sign (1 bit) | exponent (8 bits, bias 127) | mantissa (23 bits)
    // For normal numbers, mantissa has implicit leading 1
    // FP8 mantissa is 3 bits (0-7), representing fractions 0/8, 1/8, ..., 7/8
    // FP32 mantissa = (1 + mant/8 - 1) * 2^23 = mant * 2^20
    
    uint32_t fp32_exp = (uint32_t)(exponent - 7 + 127);
    uint32_t fp32_mantissa = ((uint32_t)mantissa) << 20;
    
    union { uint32_t i; float f; } u;
    u.i = (sign << 31) | (fp32_exp << 23) | fp32_mantissa;
    return u.f;
}

#ifdef __cplusplus
}
#endif

#endif // NPU_COMMON_H
