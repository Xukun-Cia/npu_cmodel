#ifndef OPERATOR_SILU_H
#define OPERATOR_SILU_H

#ifdef __cplusplus
extern "C" {
#endif

// SILU Operator
// Formula: x / (1 + exp(-x))
void user_operator_silu(int* cycle_cnt, int Rows, int Cols, float* data_ddr, float* rslt_ddr);

#ifdef __cplusplus
}
#endif

#endif // OPERATOR_SILU_H

