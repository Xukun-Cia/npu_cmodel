#ifndef T001_MATMUL_H
#define T001_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

// Test function for MatMul operator
// M, N, K: matrix dimensions
// type: data type string, must be "fp16", "bf16", or "fp8"
int test_matmul(int M, int N, int K, const char* type);

#ifdef __cplusplus
}
#endif

#endif // T001_MATMUL_H

