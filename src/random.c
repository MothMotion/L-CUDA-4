#include "config.h"
#include "random.h"

#include <stdint.h>
#include <stdlib.h>

void Randomize(arr_t* arr, const uint32_t size, const arr_t min_v, const arr_t max_v) {
  for(uint32_t i=0; i<size; ++i)
    arr[i] = rand()%(uint32_t)max_v * min_v/max_v * 10000;
    //arr[i] = MAX(min_v, rand()%(max_v+1)); 
}
