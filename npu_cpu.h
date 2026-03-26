#ifndef NPU_CPU_H
#define NPU_CPU_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU Scalar Operations
// These assume tensors are scalars (1 element)
void cpu_add(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void cpu_sub(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void cpu_mul(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);
void cpu_div(npu_tensor_t* dst, const npu_tensor_t* src1, const npu_tensor_t* src2, int* t);

#ifdef __cplusplus
}
#endif

#endif // NPU_CPU_H

