#include "config.h"
#include "random.h"

#include <stdint.h>
#include <stdlib.h>

void Randomize(arr_t** mat, const uint32_t size, const arr_t min_v, const arr_t max_v) {
  for(uint32_t i=0; i<size; ++i)
    for(uint32_t j=0; j<size; ++j)
      mat[i][j] = rand()%(uint32_t)max_v * min_v/max_v * 10000;
    //arr[i] = MAX(min_v, rand()%(max_v+1)); 
}
