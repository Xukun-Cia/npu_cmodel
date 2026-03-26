#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "operator/operator_conv3d_patched.h"
#include "t004_conv3d_patched.h"

// Helper function to parse shape from header comment
// Returns the number of dimensions, fills dims array
static int parse_shape(const char* line, int64_t* dims, int max_dims) {
    const char* start = strchr(line, '[');
    if (!start) return 0;
    start++;  // skip '['
    
    int ndim = 0;
    char* endptr;
    while (ndim < max_dims) {
        long val = strtol(start, &endptr, 10);
        if (endptr == start) break;  // no more numbers
        dims[ndim++] = val;
        start = endptr;
        while (*start == ',' || *start == ' ') start++;  // skip comma and spaces
        if (*start == ']') break;
    }
    return ndim;
}

// Helper function to load packed BF16 data from file
// File format:
//   # Shape: [dim0, dim1, ...]
//   # Dtype: bfloat16
//   # Total elements: XXX
//   # Storage: 2xBF16 packed as UINT32 ...
//   # Layout: C-contiguous ...
//   123456789  (UINT32 decimal values, one per line)
//
// Returns allocated buffer of uint16_t, sets total_elements
// Caller must free the returned buffer
static uint16_t* load_packed_bf16_file(const char* filepath, int64_t* total_elements, int64_t* shape, int* ndim) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", filepath);
        return NULL;
    }
    
    char line[256];
    *total_elements = 0;
    *ndim = 0;
    
    // Parse header comments
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') {
            // First non-comment line, rewind for data reading
            fseek(f, -strlen(line), SEEK_CUR);
            break;
        }
        
        if (strstr(line, "Shape:")) {
            *ndim = parse_shape(line, shape, MAX_DIMS + 1);
        } else if (strstr(line, "Total elements:")) {
            sscanf(line, "# Total elements: %ld", total_elements);
        }
    }
    
    if (*total_elements == 0) {
        // Calculate from shape if not found
        *total_elements = 1;
        for (int i = 0; i < *ndim; i++) {
            *total_elements *= shape[i];
        }
    }
    
    // Allocate buffer for unpacked BF16 values
    uint16_t* buffer = (uint16_t*)malloc(*total_elements * sizeof(uint16_t));
    if (!buffer) {
        fprintf(stderr, "ERROR: Cannot allocate memory for %ld elements\n", *total_elements);
        fclose(f);
        return NULL;
    }
    
    // Read packed UINT32 values and unpack to BF16
    int64_t num_uint32 = (*total_elements + 1) / 2;  // ceil division
    int64_t bf16_idx = 0;
    
    for (int64_t i = 0; i < num_uint32; i++) {
        uint32_t packed;
        if (fscanf(f, "%u", &packed) != 1) {
            fprintf(stderr, "ERROR: Failed to read UINT32 at index %ld\n", i);
            free(buffer);
            fclose(f);
            return NULL;
        }
        
        // Unpack: bf16_0 is lower 16 bits, bf16_1 is upper 16 bits
        uint16_t bf16_0 = (uint16_t)(packed & 0xFFFF);
        uint16_t bf16_1 = (uint16_t)((packed >> 16) & 0xFFFF);
        
        buffer[bf16_idx++] = bf16_0;
        if (bf16_idx < *total_elements) {
            buffer[bf16_idx++] = bf16_1;
        }
    }
    
    fclose(f);
    printf(" Loaded %s: %ld elements\n", filepath, *total_elements);
    return buffer;
}

// Test function for Conv3D operator with pre-patched input
int test_conv3d_patched(const char* data_dir) {
    printf("==============================================================\n");
    printf(" Case: CONV3D PATCHED TEST (BF16) - ViT Patch Embed from File\n");
    printf(" Precision: BF16 input, FP32 accumulator\n");
    printf("==============================================================\n");
    
    // Build file paths
    char input_path[512], weight_path[512], bias_path[512], output_path[512];
    snprintf(input_path, sizeof(input_path), "%s/conv3d_input_bf16.txt", data_dir);
    snprintf(weight_path, sizeof(weight_path), "%s/conv3d_weight_bf16.txt", data_dir);
    snprintf(bias_path, sizeof(bias_path), "%s/conv3d_bias_bf16.txt", data_dir);
    snprintf(output_path, sizeof(output_path), "%s/conv3d_output_bf16.txt", data_dir);
    
    printf("\nLoading data files from: %s\n", data_dir);
    
    // Load input data: [num_patches, C_in, T, K_H, K_W]
    int64_t input_elements, input_shape[6];
    int input_ndim;
    uint16_t* input_data = load_packed_bf16_file(input_path, &input_elements, input_shape, &input_ndim);
    if (!input_data) return 1;
    
    // Load weight data: [C_out, C_in, T, K_H, K_W]
    int64_t weight_elements, weight_shape[6];
    int weight_ndim;
    uint16_t* weight_data = load_packed_bf16_file(weight_path, &weight_elements, weight_shape, &weight_ndim);
    if (!weight_data) { free(input_data); return 1; }
    
    // Load bias data: [C_out]
    int64_t bias_elements, bias_shape[6];
    int bias_ndim;
    uint16_t* bias_data = load_packed_bf16_file(bias_path, &bias_elements, bias_shape, &bias_ndim);
    if (!bias_data) { free(input_data); free(weight_data); return 1; }
    
    // Load expected output: [num_patches, C_out, 1, 1, 1] or [num_patches, C_out]
    int64_t output_elements, output_shape[6];
    int output_ndim;
    uint16_t* expected_output = load_packed_bf16_file(output_path, &output_elements, output_shape, &output_ndim);
    if (!expected_output) { free(input_data); free(weight_data); free(bias_data); return 1; }
    
    // Extract dimensions from loaded shapes
    // Input: [num_patches, C_in, T, K_H, K_W]
    int num_patches = (int)input_shape[0];
    int C_in = (int)input_shape[1];
    int T = (int)input_shape[2];
    int K_H = (int)input_shape[3];
    int K_W = (int)input_shape[4];
    
    // Weight: [C_out, C_in, T, K_H, K_W]
    int C_out = (int)weight_shape[0];
    
    printf("\nParsed dimensions:\n");
    printf(" Input:  [%d, %d, %d, %d, %d] (num_patches, C_in, T, K_H, K_W)\n", 
           num_patches, C_in, T, K_H, K_W);
    printf(" Weight: [%d, %d, %d, %d, %d] (C_out, C_in, T, K_H, K_W)\n", 
           C_out, C_in, T, K_H, K_W);
    printf(" Bias:   [%d]\n", C_out);
    printf(" Output: [%d, %d]\n", num_patches, C_out);
    
    // Validate dimensions
    if (T != 2) {
        fprintf(stderr, "ERROR: T must be 2 (got T=%d)\n", T);
        free(input_data); free(weight_data); free(bias_data); free(expected_output);
        return 1;
    }
    
    int K = C_in * K_H * K_W;
    printf(" K dimension (C_in * K_H * K_W): %d\n", K);
    
    // Check tile alignment
    if (num_patches % 16 != 0) {
        printf(" WARNING: num_patches (%d) not divisible by 16, will use padding\n", num_patches);
    }
    if (K % 16 != 0) {
        printf(" WARNING: K (%d) not divisible by 16, will use padding\n", K);
    }
    if (C_out % 16 != 0) {
        printf(" WARNING: C_out (%d) not divisible by 16, will use padding\n", C_out);
    }
    
    // Allocate output buffer
    int output_size = num_patches * C_out;
    uint16_t* output_data = (uint16_t*)malloc(output_size * sizeof(uint16_t));
    if (!output_data) {
        fprintf(stderr, "ERROR: Cannot allocate output buffer\n");
        free(input_data); free(weight_data); free(bias_data); free(expected_output);
        return 1;
    }
    
    // Initialize
    stats_reset();
    sram_init();
    
    // ========================================
    // Run operator
    // BF16 input with FP32 accumulator (use_fp16_accu = false)
    // ========================================
    
    printf("\nRunning Conv3D Patched operator...\n");
    
    int cycles = 0;
    user_operator_conv3d_patched(&cycles, NPU_TYPE_BF16, 
                                  num_patches, C_in, T, K_H, K_W, C_out,
                                  input_data, weight_data, bias_data, output_data,
                                  false);  // use_fp16_accu = false (FP32 accumulator for BF16)
    
    // ========================================
    // Compare results with expected output
    // ========================================
    
    printf("\nChecking first 20 results:\n");
    
    float max_rel_err = 0.0f;
    float min_rel_err = 1e9f;
    double sum_rel_err = 0.0;
    int error_count = 0;
    float error_threshold = 0.1f;  // 10% relative error threshold
    
    for (int i = 0; i < output_size; i++) {
        float result = bf16_to_fp32(output_data[i]);
        float expected = bf16_to_fp32(expected_output[i]);
        float diff = fabsf(result - expected);
        float denom = fabsf(expected) > 1e-6f ? fabsf(expected) : 1e-6f;
        float rel_err = diff / denom;
        if (rel_err > 1.0f) rel_err = 1.0f;  // cap at 100%
        
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err < min_rel_err) min_rel_err = rel_err;
        sum_rel_err += rel_err;
        
        if (rel_err > error_threshold) error_count++;
        
        if (i < 20) {
            printf(" -[%d] Result: %12.6f,  Expected: %12.6f,  RelErr: %6.4f%%\n",
                   i, result, expected, rel_err * 100.0f);
        }
    }
    
    printf("\n Error Statistics (NPU vs PyTorch Reference):\n");
    printf(" - Max Relative Error: %6.4f%%\n", max_rel_err * 100.0f);
    printf(" - Min Relative Error: %6.4f%%\n", min_rel_err * 100.0f);
    printf(" - Avg Relative Error: %6.4f%%\n", (float)(sum_rel_err / output_size) * 100.0f);
    printf(" - Elements > %.0f%% error: %d / %d (%.2f%%)\n", 
           error_threshold * 100.0f, error_count, output_size, 
           (float)error_count / output_size * 100.0f);
    
    stats_print(cycles);
    
    // Cleanup
    free(input_data);
    free(weight_data);
    free(bias_data);
    free(output_data);
    free(expected_output);
    
    return 0;
}
