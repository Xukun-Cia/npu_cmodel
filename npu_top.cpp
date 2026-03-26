#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_common.h"
#include "npu_sram.h"

// Include usecase headers
#include "t000_silu.h"
#include "t001_matmul.h"
#include "t002_gelu.h"
#include "t003_conv3d.h"
#include "t004_conv3d_patched.h"

// Global Stats Definition
// {total_inst, inst_cpu, inst_rvv, inst_tpu, inst_dma, inst_sram}
npu_stats_t g_npu_stats = {0, 0, 0, 0, 0, 0};

// ==========================================
// MAIN FOR TESTING
// ==========================================

void print_usage(const char* prog_name) {
    printf("Usage: %s -t <test_case> [options]\n", prog_name);
    printf("\nAvailable test cases (can use short form like t000, t001, etc.):\n");
    printf("  t000  (t000_silu)           - SILU operator test (FP32)\n");
    printf("  t001  (t001_matmul)         - MatMul operator test (FP16/BF16/FP8)\n");
    printf("  t002  (t002_gelu)           - GELU operator test (FP32)\n");
    printf("  t003  (t003_conv3d)         - Conv3D operator test (FP16/BF16) - ViT Patch Embed\n");
    printf("  t004  (t004_conv3d_patched) - Conv3D patched test (BF16) - Load from data files\n");
    printf("\nOptions for t000 (SILU):\n");
    printf("  --rows <num>       - Number of rows (default: 512)\n");
    printf("  --cols <num>       - Number of columns (default: 9728)\n");
    printf("\nOptions for t001 (MatMul):\n");
    printf("  --M <num>          - Matrix M dimension (default: 512)\n");
    printf("  --N <num>          - Matrix N dimension (default: 128)\n");
    printf("  --K <num>          - Matrix K dimension (default: 80)\n");
    printf("  --type <type>      - Data type: fp16, bf16, or fp8 (default: fp16)\n");
    printf("\nOptions for t002 (GELU):\n");
    printf("  --rows <num>       - Number of rows (default: 1024)\n");
    printf("  --cols <num>       - Number of columns (default: 4096)\n");
    printf("\nOptions for t003 (Conv3D - Qwen3-VL-4B ViT Patch Embed):\n");
    printf("  --H <num>          - Input height (default: 448)\n");
    printf("  --W <num>          - Input width (default: 448)\n");
    printf("  --C_in <num>       - Input channels (default: 3)\n");
    printf("  --T <num>          - Temporal dimension (default: 2)\n");
    printf("  --K_H <num>        - Kernel height (default: 16)\n");
    printf("  --K_W <num>        - Kernel width (default: 16)\n");
    printf("  --C_out <num>      - Output channels (default: 1024)\n");
    printf("  --type <type>      - Data type: fp16 or bf16 (default: bf16)\n");
    printf("\nOptions for t004 (Conv3D Patched - BF16 input, FP32 accumulator):\n");
    printf("  --data_dir <path>  - Path to data directory (default: ../data)\n");
    printf("\nExamples:\n");
    printf("  %s -t t000\n", prog_name);
    printf("  %s -t t000 --rows 256 --cols 4096\n", prog_name);
    printf("  %s -t t001\n", prog_name);
    printf("  %s -t t001 --M 512 --N 128 --K 80 --type fp16\n", prog_name);
    printf("  %s -t t001 --type bf16\n", prog_name);
    printf("  %s -t t002\n", prog_name);
    printf("  %s -t t002 --rows 1024 --cols 4096\n", prog_name);
    printf("  %s -t t003\n", prog_name);
    printf("  %s -t t003 --type fp16\n", prog_name);
    printf("  %s -t t003 --H 448 --W 448 --C_out 1024 --type bf16\n", prog_name);
    printf("  %s -t t004\n", prog_name);
    printf("  %s -t t004 --data_dir ../data\n", prog_name);
}

int parse_int_arg(const char* arg_name, const char* value, int* result) {
    char* endptr;
    long val = strtol(value, &endptr, 10);
    if (*endptr != '\0' || val <= 0) {
        fprintf(stderr, "ERROR: Invalid %s value '%s'. Must be a positive integer.\n", arg_name, value);
        return 1;
    }
    *result = (int)val;
    return 0;
}

// Helper to match test case name (matches first 4 characters)
static bool match_test_case(const char* input, const char* test_id) {
    return strncmp(input, test_id, 4) == 0;
}

int main(int argc, char* argv[]) {
    printf("==============================================================\n");
    printf(" NPU C-Model Simulator\n");
    printf("==============================================================\n");
    
    // Parse command line arguments
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (strcmp(argv[1], "-t") != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* test_case = argv[2];
    
    // Route to appropriate test case (match first 4 characters)
    int result = 0;
    if (match_test_case(test_case, "t000")) {
        // Default values for SILU
        int rows = 512;
        int cols = 9728;
        
        // Parse optional arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--rows") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --rows requires a value\n");
                    return 1;
                }
                if (parse_int_arg("rows", argv[++i], &rows) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--cols") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --cols requires a value\n");
                    return 1;
                }
                if (parse_int_arg("cols", argv[++i], &cols) != 0) {
                    return 1;
                }
            } else {
                fprintf(stderr, "ERROR: Unknown option '%s' for t000\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
        
        result = test_silu(rows, cols);
        
    } else if (match_test_case(test_case, "t001")) {
        // Default values for MatMul
        int M = 512;
        int N = 128;
        int K = 80;
        const char* type = "fp16";
        
        // Parse optional arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--M") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --M requires a value\n");
                    return 1;
                }
                if (parse_int_arg("M", argv[++i], &M) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--N") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --N requires a value\n");
                    return 1;
                }
                if (parse_int_arg("N", argv[++i], &N) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--K") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --K requires a value\n");
                    return 1;
                }
                if (parse_int_arg("K", argv[++i], &K) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--type") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --type requires a value\n");
                    return 1;
                }
                type = argv[++i];
                if (strcmp(type, "fp16") != 0 && strcmp(type, "bf16") != 0 && strcmp(type, "fp8") != 0) {
                    fprintf(stderr, "ERROR: Invalid type '%s'. Must be 'fp16', 'bf16', or 'fp8'\n", type);
                    return 1;
                }
            } else {
                fprintf(stderr, "ERROR: Unknown option '%s' for t001\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
        
        result = test_matmul(M, N, K, type);
        
    } else if (match_test_case(test_case, "t002")) {
        // Default values for GELU
        int rows = 1024;
        int cols = 4096;
        
        // Parse optional arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--rows") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --rows requires a value\n");
                    return 1;
                }
                if (parse_int_arg("rows", argv[++i], &rows) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--cols") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --cols requires a value\n");
                    return 1;
                }
                if (parse_int_arg("cols", argv[++i], &cols) != 0) {
                    return 1;
                }
            } else {
                fprintf(stderr, "ERROR: Unknown option '%s' for t002\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
        
        result = test_gelu(rows, cols);
        
    } else if (match_test_case(test_case, "t003")) {
        // Default values for Conv3D (Qwen3-VL-4B ViT Patch Embed)
        int H = 448;
        int W = 448;
        int C_in = 3;
        int T = 2;
        int K_H = 16;
        int K_W = 16;
        int C_out = 1024;
        const char* type = "bf16";
        
        // Parse optional arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--H") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --H requires a value\n");
                    return 1;
                }
                if (parse_int_arg("H", argv[++i], &H) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--W") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --W requires a value\n");
                    return 1;
                }
                if (parse_int_arg("W", argv[++i], &W) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--C_in") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --C_in requires a value\n");
                    return 1;
                }
                if (parse_int_arg("C_in", argv[++i], &C_in) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--T") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --T requires a value\n");
                    return 1;
                }
                if (parse_int_arg("T", argv[++i], &T) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--K_H") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --K_H requires a value\n");
                    return 1;
                }
                if (parse_int_arg("K_H", argv[++i], &K_H) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--K_W") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --K_W requires a value\n");
                    return 1;
                }
                if (parse_int_arg("K_W", argv[++i], &K_W) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--C_out") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --C_out requires a value\n");
                    return 1;
                }
                if (parse_int_arg("C_out", argv[++i], &C_out) != 0) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--type") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --type requires a value\n");
                    return 1;
                }
                type = argv[++i];
                if (strcmp(type, "fp16") != 0 && strcmp(type, "bf16") != 0) {
                    fprintf(stderr, "ERROR: Invalid type '%s'. Must be 'fp16' or 'bf16'\n", type);
                    return 1;
                }
            } else {
                fprintf(stderr, "ERROR: Unknown option '%s' for t003\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
        
        result = test_conv3d(H, W, C_in, T, K_H, K_W, C_out, type);
        
    } else if (match_test_case(test_case, "t004")) {
        // Default values for Conv3D Patched (BF16 with FP32 accumulator)
        const char* data_dir = "../data";
        
        // Parse optional arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--data_dir") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "ERROR: --data_dir requires a value\n");
                    return 1;
                }
                data_dir = argv[++i];
            } else {
                fprintf(stderr, "ERROR: Unknown option '%s' for t004\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
        
        result = test_conv3d_patched(data_dir);
        
    } else {
        printf("ERROR: Unknown test case: %s\n\n", test_case);
        print_usage(argv[0]);
        return 1;
    }
    
    printf(" Simulation Done\n");
    printf("==============================================================\n\n");

    return result;
}
