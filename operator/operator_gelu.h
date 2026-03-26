#ifndef OPERATOR_GELU_H
#define OPERATOR_GELU_H

#ifdef __cplusplus
extern "C" {
#endif

// GELU Operator
// Formula: 0.5 * x * (1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)))
void user_operator_gelu(int* cycle_cnt, int Rows, int Cols, float* data_ddr, float* rslt_ddr);

#ifdef __cplusplus
}
#endif

#endif // OPERATOR_GELU_H

