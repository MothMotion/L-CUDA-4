#ifndef SERIAL



#include "config.h"
#include "matrix.h"
#include "timer.h"
#include "vec_oper.h"
#include "k_vec_oper.h"

#include <stdint.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime.h>



time_s Operation(const matrix& arr1, const matrix& arr2, matrix& out, const Oper& op) {
  time_s time;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  arr_t *d_arr1, *d_arr2, *d_out;\
  cudaMalloc((void**)&d_arr1, arr1.size*arr1.size*sizeof(arr_t));\
  cudaMalloc((void**)&d_arr2, arr2.size*arr2.size*sizeof(arr_t));\
  cudaMalloc((void**)&d_out, out.size*out.size*sizeof(arr_t));\

  float *h_arr1, *h_arr2, *flat_out;
  cudaStream_t stream;
  cudaStreamCreate(&stream); 


  CUDATIME(({
    h_arr1 = arr1.flat();
    h_arr2 = arr2.flat();
    flat_out = (arr_t*)malloc(out.size*out.size*sizeof(arr_t));

    cudaHostRegister(h_arr1, arr1.size*arr1.size, cudaHostRegisterDefault);
    cudaHostRegister(h_arr2, arr2.size*arr2.size, cudaHostRegisterDefault);
    cudaHostRegister(flat_out, out.size*out.size, cudaHostRegisterDefault);

    cudaMemcpyAsync(d_arr1, h_arr1, arr1.size*arr1.size*sizeof(arr_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_arr2, h_arr2, arr2.size*arr2.size*sizeof(arr_t), cudaMemcpyHostToDevice, stream);
  }), time.memcpy, start, end);

  cudaHostUnregister(h_arr1);
  free(h_arr1);
  cudaHostUnregister(h_arr2);
  free(h_arr2);

  dim3 blocks(KBLOCKS, 1, 1);
  dim3 threads(KTHREADS, 1, 1);

  CUDATIME(({
    switch(op) {
      case opadd : KAdd<<<blocks,threads>>>(d_arr1, d_arr2, d_out, out.size*out.size); break;
      case opsub : KSub<<<blocks,threads>>>(d_arr1, d_arr2, d_out, out.size*out.size); break;
      case opmul : KMul<<<blocks,threads>>>(d_arr1, d_arr2, d_out, out.size*out.size); break;
      case opdiv : KDiv<<<blocks,threads>>>(d_arr1, d_arr2, d_out, out.size*out.size); break;
      default : break;
    } 
  }), time.run, start, end);

  CUDATIME(({
    cudaMemcpyAsync(flat_out, d_out, size, cudaMemcpyDeviceToHost, stream);
    cudaHostUnregister(flat_out);
    out.deflat(flat_out);
  }), time.memret, start, end);

  time.total = time.memcpy + time.run + time.memret;

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(d_arr1);
  cudaFree(d_arr2);
  cudaFree(d_out);

  return time;
}



__global__ void KAdd(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] + arr2[idx];
}

__global__ void KSub(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] - arr2[idx];
}

__global__ void KMul(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
    out[idx] = arr1[idx] * arr2[idx];
}

__global__ void KDiv(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size && arr2[idx])
    out[idx] = arr1[idx] / arr2[idx];
}

#endif
