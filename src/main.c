#include "config.h"
#include "random.h"
#include "timer.h"
#include "vec_oper.h"

#ifndef SERIAL
#include "k_vec_wrap.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>



int main() {
  time_s time_temp;
  uint32_t arr_size = ARRAY_SIZE,
           cycles   = CYCLES;

  arr_t *arr1 = malloc(arr_size * sizeof(arr_t)),
        *arr2 = malloc(arr_size * sizeof(arr_t)),
        *out  = malloc(arr_size * sizeof(arr_t));

  float avg_rand = 0;

  time_s avg_time[4];
  for(enum Oper i = opadd; i <= opdiv; ++i)
    EMPTYTIME(avg_time[i]);

  #ifdef SERIAL
  printf("Serial ");
  #else
  printf("Parallel ");
  #endif
  printf("execution.\nSettings:\n\tArray size:\t%d\n\tCycles:\t\t%d\n\tBlocks:\t\t%d\n\tThreads:\t%d\n\tElement size:\t%d\n\n",
         arr_size, cycles, KBLOCKS, KTHREADS, (uint32_t)sizeof(arr_t));



  for(uint32_t i=0; i<cycles; ++i) {
    GETTIME(({
      Randomize(arr1, arr_size, MIN_RAND, MAX_RAND);
      Randomize(arr2, arr_size, MIN_RAND, MAX_RAND);
    }), time_temp.total);
    avg_rand += time_temp.total/cycles;

    for(enum Oper i=opadd; i<=opdiv; ++i) {
      time_temp = Operation(arr1, arr2, out, arr_size, i);
      time_div(&time_temp, cycles);
      time_add(&avg_time[i], &time_temp);
    }
  }

  printf("Average time spent per cycle.\nRandomizing:\t%f\n", avg_rand);
  for(enum Oper i=opadd; i<=opdiv; ++i) {
    char* text;
    switch(i) {
      case opadd : text = "Summation"; break;
      case opsub : text = "Substraction"; break;
      case opmul : text = "Multiplication"; break;
      case opdiv : text = "Division"; break;
    }
    printf("%s:\n", text);

    #ifdef SERIAL
    PRINTTIME(avg_time[i], "Total:\t%f\n\n");

    #else
    PRINTTIME(avg_time[i], "Total:\t%f\n", "\tCopying:\t%f\n", "\tRunning:\t%f\n", "\tReturning:\t%f\n\n");

    #endif
  }

  return 0;
}
