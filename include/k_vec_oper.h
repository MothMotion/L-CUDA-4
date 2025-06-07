#pragma once
#if !defined(K_VEC_OPER_H) && !defined(SERIAL)
#define K_VEC_OPER_H



#include "config.h"
#include "timer.h"
#include "vec_oper.h"

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>



__global__ void KAdd(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size);
__global__ void KSub(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size);
__global__ void KMul(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size);
__global__ void KDiv(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size);

#endif
