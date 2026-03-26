#ifndef OPERATOR_MATMUL_H
#define OPERATOR_MATMUL_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// MatMul Operator
// D = A * B
void user_operator_matmul(int* cycle_cnt, npu_datatype_t type, int M, int N, int K, void* a_ddr, void* b_ddr, void* rslt_ddr);

#ifdef __cplusplus
}
#endif

#endif // OPERATOR_MATMUL_H

