#ifndef SERIAL



#include "config.h"
#include "timer.h"
#include "vec_oper.h"
#include "k_vec_oper.h"

#include <stdint.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime.h>



extern "C" time_s Operation(arr_t** arr1, arr_t** arr2, arr_t** out, const uint32_t size, const enum Oper op) {
  time_s time;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  arr_t *d_arr1, *d_arr2, *d_out;\
  cudaMalloc((void**)&d_arr1, size*sizeof(arr_t));\
  cudaMalloc((void**)&d_arr2, size*sizeof(arr_t));\
  cudaMalloc((void**)&d_out, size*sizeof(arr_t));\


  CUDATIME(({
    cudaMemcpy(d_arr1, arr1, size*sizeof(arr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, size*sizeof(arr_t), cudaMemcpyHostToDevice); 
  }), time.memcpy, start, end);

  CUDATIME(({
    switch(op) {
      case opadd : KAdd<<<KBLOCKS,KTHREADS>>>(d_arr1, d_arr2, d_out, size); break;
      case opsub : KSub<<<KBLOCKS,KTHREADS>>>(d_arr1, d_arr2, d_out, size); break;
      case opmul : KMul<<<KBLOCKS,KTHREADS>>>(d_arr1, d_arr2, d_out, size); break;
      case opdiv : KDiv<<<KBLOCKS,KTHREADS>>>(d_arr1, d_arr2, d_out, size); break;
    } 
  }), time.run, start, end);

  CUDATIME(({
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
  }), time.memret, start, end);

  time.total = time.memcpy + time.run + time.memret;

  cudaFree(d_arr1);
  cudaFree(d_arr2);
  cudaFree(d_out);

  return time;
}

__global__ void KAdd(arr_t* arr1, arr_t* arr2, arr_t* out, uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] + arr2[idx];
}

__global__ void KSub(arr_t* arr1, arr_t* arr2, arr_t* out, uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] - arr2[idx];
}

__global__ void KMul(arr_t* arr1, arr_t* arr2, arr_t* out, uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] * arr2[idx];
}

__global__ void KDiv(arr_t* arr1, arr_t* arr2, arr_t* out, uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size && arr2[idx])
    out[idx] = arr1[idx] / arr2[idx];
}

#endif
