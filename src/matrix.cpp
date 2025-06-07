#include "config.h"
#include "matrix.h"

#include "stdint.h"
#include "stdlib.h"



arr_t* matrix::flat() const {
  float* temp = (float*)malloc(size*size * sizeof(arr_t));
  for(uint32_t i=0; i<size; ++i)
    for(uint32_t j=0; j<size; ++j)
      temp[i*size + j] = data[i][j];
  return temp;
}

void matrix::deflat(arr_t* from) {
  for(uint32_t i=0; i<size; ++i)
    for(uint32_t j=0; j<size; ++j)
      data[i][j] = from[i*size + j];
  free(from);
}

void matrix::init() {
  data = (float**)malloc(size * sizeof(arr_t*));
  for(uint32_t i=0; i<size; ++i)
    data[i] = (float*)malloc(size * sizeof(arr_t));
}

void matrix::deinit() {
  for(uint32_t i=0; i<size; ++i) {
    free(data[i]);
    data[i] = nullptr;
  }
  free(data);
  data = nullptr;
}
