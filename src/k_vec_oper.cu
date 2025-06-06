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

  arr_t *h_arr1 = (float*)malloc(size*size*sizeof(arr_t)),
        *h_arr2 = (float*)malloc(size*size*sizeof(arr_t)),
        *h_out  = (float*)malloc(size*size*sizeof(arr_t));

  GETTIME(({
    Flat(arr1, h_arr1, size);
    Flat(arr2, h_arr2, size);
  }), time.flat);

  arr_t *d_arr1, *d_arr2, *d_out;\
  cudaMalloc((void**)&d_arr1, size*size*sizeof(arr_t));\
  cudaMalloc((void**)&d_arr2, size*size*sizeof(arr_t));\
  cudaMalloc((void**)&d_out, size*size*sizeof(arr_t));\


  CUDATIME(({
    cudaMemcpy(d_arr1, h_arr1, size*sizeof(arr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, size*sizeof(arr_t), cudaMemcpyHostToDevice); 
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
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  }), time.memret, start, end);

  GETTIME(({
    Deflat(h_out, out, size);
  }), time.total);
  time.flat += time.total;
  time.memcpy /= 1000;
  time.run /= 1000;
  time.memret /= 1000;

  time.total = time.flat + time.memcpy + time.run + time.memret;

  cudaFree(d_arr1);
  cudaFree(d_arr2);
  cudaFree(d_out);

  return time;
}

void Flat(arr_t** matrix, arr_t* array, const uint32_t size) {
  for(uint32_t i=0; i<size; ++i)
    for(uint32_t j=0; i<size; ++j)
      array[i*size + j] = matrix[i][j];
}

void Deflat(arr_t* array, arr_t** matrix, const uint32_t size) {
  for(uint32_t i=0; i<size; ++i)
    for(uint32_t j=0; j<size; ++j)
      matrix[i][j] = array[i*size + j];
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
