#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "operator/operator_conv3d.h"
#include "t003_conv3d.h"

// Test function for Conv3D operator (ViT Patch Embed)
// Output: [(T/2) * num_patches, C_out] - every 2 frames produce one result
// Precision: FP32 accumulator for all input types
int test_conv3d(int H, int W, int C_in, int T, int K_H, int K_W, int C_out, const char* type) {
    // Validate type parameter
    npu_datatype_t npu_type;
    if (strcmp(type, "fp16") == 0) {
        npu_type = NPU_TYPE_FP16;
        printf(" Case: CONV3D TEST (FP16 input, FP32 accumulator) - ViT Patch Embed\n");
    } else if (strcmp(type, "bf16") == 0) {
        npu_type = NPU_TYPE_BF16;
        printf(" Case: CONV3D TEST (BF16 input, FP32 accumulator) - ViT Patch Embed\n");
    } else {
        fprintf(stderr, "ERROR: Invalid type '%s'. Must be 'fp16' or 'bf16'\n", type);
        return 1;
    }
    
    printf("==============================================================\n");
    //printf(" Input: [%d, %d, %d, %d] (T, H, W, C_in)\n", T, H, W, C_in);
    //printf(" Kernel: [%d, %d, 2, %d, %d] (C_out, C_in, T_kernel, K_H, K_W)\n", C_out, C_in, K_H, K_W);
    
    // Validate dimensions
    if (T % 2 != 0) {
        fprintf(stderr, "ERROR: T(%d) must be even\n", T);
        return 1;
    }
    if (H % K_H != 0 || W % K_W != 0) {
        fprintf(stderr, "ERROR: H(%d) must be divisible by K_H(%d), W(%d) must be divisible by K_W(%d)\n", 
                H, K_H, W, K_W);
        return 1;
    }
    
    int num_patches = (H / K_H) * (W / K_W);
    int kernel_elements = K_H * K_W * C_in;
    int num_frame_pairs = T / 2;
    int total_output_patches = num_frame_pairs * num_patches;
    
    //printf(" Output: [%d, %d] (total_patches = %d frame_pairs * %d patches, C_out)\n", 
    //       total_output_patches, C_out, num_frame_pairs, num_patches);
    //printf(" Each frame pair: matmul [%d, %d] x [%d, %d] x 2\n", 
    //       num_patches, kernel_elements, kernel_elements, C_out);
    
    // Validate tile alignment
    if (num_patches % 16 != 0 || kernel_elements % 16 != 0 || C_out % 16 != 0) {
        fprintf(stderr, "ERROR: Dimensions must be divisible by 16 for tiled computation\n");
        fprintf(stderr, "       num_patches=%d, kernel_elements=%d, C_out=%d\n", 
                num_patches, kernel_elements, C_out);
        return 1;
    }
    
    stats_reset();
    sram_init();
    
    unsigned int seed = time(NULL);
    
    // Calculate sizes
    int input_frame_size = H * W * C_in;
    int input_total_size = T * input_frame_size;
    int weight_size = C_out * C_in * 2 * K_H * K_W;  // Weight T dimension is always 2
    int output_size = total_output_patches * C_out;
    
    // Allocate memory
    uint16_t* input_data = (uint16_t*)malloc(input_total_size * sizeof(uint16_t));
    uint16_t* weight_data = (uint16_t*)malloc(weight_size * sizeof(uint16_t));
    uint16_t* bias_data = (uint16_t*)malloc(C_out * sizeof(uint16_t));
    uint16_t* output_data = (uint16_t*)malloc(output_size * sizeof(uint16_t));
    uint16_t* output_ref = (uint16_t*)malloc(output_size * sizeof(uint16_t));
    
    if (!input_data || !weight_data || !bias_data || !output_data || !output_ref) {
        printf("Failed to allocate memory\n");
        return 1;
    }
    
    // Helper functions for current precision
    auto to_fp32 = [npu_type](uint16_t val) -> float {
        return (npu_type == NPU_TYPE_FP16) ? fp16_to_fp32(val) : bf16_to_fp32(val);
    };
    auto from_fp32 = [npu_type](float val) -> uint16_t {
        return (npu_type == NPU_TYPE_FP16) ? fp32_to_fp16(val) : fp32_to_bf16(val);
    };
    
    // Generate random data based on type
    if (npu_type == NPU_TYPE_FP16) {
        auto generate_fp16_random = [&seed]() -> float {
            uint8_t sign = rand_r(&seed) & 0x1;
            uint8_t exp = 10 + (rand_r(&seed) % 10);  // Moderate range to avoid overflow
            uint16_t mant = rand_r(&seed) & 0x3FF;
            uint16_t fp16_val = (sign << 15) | (exp << 10) | mant;
            return fp16_to_fp32(fp16_val);
        };
        
        // Generate single frame and duplicate to T frames
        for (int i = 0; i < input_frame_size; i++) {
            float val = generate_fp16_random();
            uint16_t fp16_val = fp32_to_fp16(val);
            for (int t = 0; t < T; t++) {
                input_data[t * input_frame_size + i] = fp16_val;
            }
        }
        
        // Generate weight
        for (int i = 0; i < weight_size; i++) {
            weight_data[i] = fp32_to_fp16(generate_fp16_random());
        }
        
        // Generate bias
        for (int i = 0; i < C_out; i++) {
            bias_data[i] = fp32_to_fp16(generate_fp16_random());
        }
        
    } else { // BF16
        auto generate_bf16_random = [&seed]() -> float {
            uint8_t sign = rand_r(&seed) & 0x1;
            uint8_t exp = 120 + (rand_r(&seed) % 15);  // Moderate range
            uint8_t mant = rand_r(&seed) & 0x7F;
            uint16_t bf16_val = (sign << 15) | (exp << 7) | mant;
            return bf16_to_fp32(bf16_val);
        };
        
        // Generate single frame and duplicate to T frames
        for (int i = 0; i < input_frame_size; i++) {
            float val = generate_bf16_random();
            uint16_t bf16_val = fp32_to_bf16(val);
            for (int t = 0; t < T; t++) {
                input_data[t * input_frame_size + i] = bf16_val;
            }
        }
        
        // Generate weight
        for (int i = 0; i < weight_size; i++) {
            weight_data[i] = fp32_to_bf16(generate_bf16_random());
        }
        
        // Generate bias
        for (int i = 0; i < C_out; i++) {
            bias_data[i] = fp32_to_bf16(generate_bf16_random());
        }
    }
    
    // ========================================
    // Calculate reference result (FP32 accumulator)
    // ========================================
    
    printf("\nComputing reference result (FP32 accumulator)...\n");
    
    // Reshape weights for reference (same as operator)
    uint16_t* weight_b_ref[2];
    for (int t = 0; t < 2; t++) {
        weight_b_ref[t] = (uint16_t*)malloc(kernel_elements * C_out * sizeof(uint16_t));
        
        for (int oc = 0; oc < C_out; oc++) {
            for (int ic = 0; ic < C_in; ic++) {
                for (int kh = 0; kh < K_H; kh++) {
                    for (int kw = 0; kw < K_W; kw++) {
                        int src_idx = oc * (C_in * 2 * K_H * K_W) +
                                      ic * (2 * K_H * K_W) +
                                      t * (K_H * K_W) +
                                      kh * K_W + kw;
                        int row = kh * K_W * C_in + kw * C_in + ic;
                        int dst_idx = row * C_out + oc;
                        weight_b_ref[t][dst_idx] = weight_data[src_idx];
                    }
                }
            }
        }
    }
    
    int num_patches_h = H / K_H;
    int num_patches_w = W / K_W;
    int tile_M = num_patches / 16;
    int tile_N = C_out / 16;
    int tile_K = kernel_elements / 16;
    
    // Process each frame pair
    for (int fp = 0; fp < num_frame_pairs; fp++) {
        int frame_0 = fp * 2;
        int frame_1 = fp * 2 + 1;
        int output_offset = fp * num_patches;
        
        // im2col for both frames
        uint16_t* im2col_ref[2];
        for (int t = 0; t < 2; t++) {
            im2col_ref[t] = (uint16_t*)malloc(num_patches * kernel_elements * sizeof(uint16_t));
            
            int frame_idx = (t == 0) ? frame_0 : frame_1;
            uint16_t* frame_ptr = input_data + frame_idx * input_frame_size;
            
            for (int ph = 0; ph < num_patches_h; ph++) {
                for (int pw = 0; pw < num_patches_w; pw++) {
                    int patch_idx = ph * num_patches_w + pw;
                    int src_h_start = ph * K_H;
                    int src_w_start = pw * K_W;
                    
                    int dst_col = 0;
                    for (int kh = 0; kh < K_H; kh++) {
                        for (int kw = 0; kw < K_W; kw++) {
                            for (int c = 0; c < C_in; c++) {
                                int src_h = src_h_start + kh;
                                int src_w = src_w_start + kw;
                                int src_idx = src_h * W * C_in + src_w * C_in + c;
                                int dst_idx = patch_idx * kernel_elements + dst_col;
                                im2col_ref[t][dst_idx] = frame_ptr[src_idx];
                                dst_col++;
                            }
                        }
                    }
                }
            }
        }
        
        // Allocate FP32 accumulator for this frame pair
        float* rslt_fp32 = (float*)malloc(num_patches * C_out * sizeof(float));
        
        // Initialize with bias (converted to FP32)
        for (int m = 0; m < num_patches; m++) {
            for (int n = 0; n < C_out; n++) {
                rslt_fp32[m * C_out + n] = to_fp32(bias_data[n]);
            }
        }
        
        // Simulate tiled matmul for both frames with FP32 accumulator
        for (int t = 0; t < 2; t++) {
            for (int m_idx = 0; m_idx < tile_M; m_idx++) {
                for (int n_idx = 0; n_idx < tile_N; n_idx++) {
                    for (int k_idx = 0; k_idx < tile_K; k_idx++) {
                        // Process 16x16 tile
                        for (int m = 0; m < 16; m++) {
                            for (int n = 0; n < 16; n++) {
                                int r = m_idx * 16 + m;
                                int c = n_idx * 16 + n;
                                float sum = rslt_fp32[r * C_out + c];
                                
                                for (int k = 0; k < 16; k++) {
                                    int k_global = k_idx * 16 + k;
                                    float a = to_fp32(im2col_ref[t][r * kernel_elements + k_global]);
                                    float b = to_fp32(weight_b_ref[t][k_global * C_out + c]);
                                    sum += a * b;
                                }
                                
                                rslt_fp32[r * C_out + c] = sum;
                            }
                        }
                    }
                }
            }
        }
        
        // Convert FP32 result back to original type
        for (int m = 0; m < num_patches; m++) {
            for (int n = 0; n < C_out; n++) {
                output_ref[(output_offset + m) * C_out + n] = from_fp32(rslt_fp32[m * C_out + n]);
            }
        }
        
        free(rslt_fp32);
        
        // Cleanup im2col buffers
        for (int t = 0; t < 2; t++) {
            free(im2col_ref[t]);
        }
    }
    
    // Cleanup weight reshape buffers
    for (int t = 0; t < 2; t++) {
        free(weight_b_ref[t]);
    }
    
    // ========================================
    // Run operator
    // ========================================
    
    printf("Running Conv3D operator...\n");
    
    int cycles = 0;
    user_operator_conv3d(&cycles, npu_type, H, W, C_in, T, K_H, K_W, C_out,
                         input_data, weight_data, bias_data, output_data);
    
    // ========================================
    // Compare results
    // ========================================
    
    printf("\nChecking first 10 results:\n");
    
    float max_rel_err = 0.0f;
    float min_rel_err = 1e9f;
    double sum_rel_err = 0.0;
    
    for (int i = 0; i < output_size; i++) {
        float result = to_fp32(output_data[i]);
        float expected = to_fp32(output_ref[i]);
        float diff = fabsf(result - expected);
        float rel_err = (fabsf(expected) > 1e-3f) ? diff / fabsf(expected) : diff / 1e-3f;
        if (rel_err > 1.0f) rel_err = 1.0f;
        
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err < min_rel_err) min_rel_err = rel_err;
        sum_rel_err += rel_err;
        
        if (i < 10) {
            printf(" -[%d] Result: %16.4f,  Expected: %16.4f,  RelErr: %4.4f%%\n",
                   i, result, expected, rel_err * 100.0f);
        }
    }
    
    printf("\n Error Statistics:\n - Max: %4.4f%%\n - Min: %4.4f%%\n - Avg: %4.4f%%\n\n",
           max_rel_err * 100.0f, min_rel_err * 100.0f, (float)(sum_rel_err / output_size) * 100.0f);
    
    stats_print(cycles);
    
    // Cleanup
    free(input_data);
    free(weight_data);
    free(bias_data);
    free(output_data);
    free(output_ref);
    
    return 0;
}

