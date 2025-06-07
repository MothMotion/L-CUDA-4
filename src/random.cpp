#include "config.h"
#include "matrix.h"
#include "random.h"

#include <stdint.h>
#include <stdlib.h>



void Randomize(const matrix& mat, const arr_t& min_v, const arr_t& max_v) {
  for(uint32_t i=0; i<mat.size; ++i)
    for(uint32_t j=0; j<mat.size; ++j)
      mat.data[i][j] = rand() * (max_v - min_v)/(RAND_MAX - min_v) + min_v;
    //arr[i] = MAX(min_v, rand()%(max_v+1)); 
}
