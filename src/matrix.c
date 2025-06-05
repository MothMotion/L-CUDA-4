#include "config.h"
#include "matrix.h"

#include <stdlib.h>



void Init(arr_t** mat, const uint32_t size) {
  for(uint32_t i=0; i<size; ++i)
    mat[i] = malloc(size * sizeof(arr_t));
}

void Deinit(arr_t** mat, const uint32_t size) {
  for(uint32_t i=0; i<size; ++i) {
    free(mat[i]);
    mat[i] = NULL;
  }
  free(mat);
}
