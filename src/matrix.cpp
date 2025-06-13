#include "config.h"
#include "matrix.h"

#include "stdint.h"
#include "stdlib.h"
#include <cstring>



void matrix::flat(arr_t* to) const { 
  for(uint32_t i=0; i<size; ++i)
    memcpy(&to[i*size], data[i], sizeof(arr_t));
}

void matrix::deflat(arr_t* from) {
  for(uint32_t i=0; i<size; ++i)
    memcpy(data[i], &from[i*size], sizeof(arr_t));
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
