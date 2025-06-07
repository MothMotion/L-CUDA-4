#include "config.h"
#include "random.h"
#include "timer.h"
#include "vec_oper.h"

#ifndef SERIAL
#include "k_vec_oper.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>



int main() {
  time_s time_temp;
  uint32_t arr_size = ARRAY_SIZE,
           cycles   = CYCLES;

  matrix mat1 = {nullptr, arr_size},
         mat2 = {nullptr, arr_size},
         out  = {nullptr, arr_size};

  mat1.init();
  mat2.init();
  out.init();

  float avg_rand = 0;

  time_s avg_time[Oper::size]; 

  #ifdef SERIAL
  printf("Serial ");
  #else
  printf("Parallel ");
  #endif
  printf("execution.\nSettings:\n\tMaxtrix size:\t%d\n\tCycles:\t\t%d\n\tBlocks:\t\t%d\n\tThreads:\t%d\n\tElement size:\t%d\n\n",
         arr_size, cycles, ( (KBLOCKS) ? (KBLOCKS) : ((arr_size%KTHREADS) ? (arr_size/KTHREADS + 1) : (arr_size/KTHREADS)) ), KTHREADS, (uint32_t)sizeof(arr_t));



  for(uint32_t i=0; i<cycles; ++i) {
    GETTIME(({
      Randomize(mat1, MIN_RAND, MAX_RAND);
      Randomize(mat2, MIN_RAND, MAX_RAND);
    }), time_temp.total);
    avg_rand += time_temp.total/cycles * 1000;

    for(uint8_t i=Oper::opadd; i<Oper::size; ++i) 
      avg_time[i] += Operation(mat1, mat2, out, (Oper)i) / cycles;
  }

  printf("Average time spent per cycle.\nRandomizing:\t%fms\n", avg_rand);
  for(uint8_t i=Oper::opadd; i<Oper::size; ++i) {
    char* text;
    switch(i) {
      case opadd : text = (char*)"\n\nSummation"; break;
      case opsub : text = (char*)"\n\nSubstraction"; break;
      case opmul : text = (char*)"\n\nMultiplication"; break;
      case opdiv : text = (char*)"\n\nDivision"; break;
      default : text = (char*)"\n\nUnknown"; break;
    }
    printf("%s:\n", text);

    #ifdef SERIAL
    avg_time[i].print("Total:\t");

    #else
    avg_time[i].print("Total:\t", "ms\n\tCopying:\t", "ms\n\tRunning:\t", "ms\n\tReturning:\t");

    #endif
    printf("ms\n");
  }

  mat1.deinit();
  mat2.deinit();
  out.deinit();

  return 0;
}
