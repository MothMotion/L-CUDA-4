#pragma once
#ifndef TIMER
#define TIMER



#include "config.h"
#include <time.h>



#define GETTIME(code_block, time_var) {\
  struct timespec start, end;\
  clock_gettime(CLOCK_REALTIME, &start);\
  code_block;\
  clock_gettime(CLOCK_REALTIME, &end);\
  time_var = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/1e9;\
}

#define CUDATIME(code_block, time_var, start, end)\
  cudaEventRecord(start);\
  code_block;\
  cudaEventRecord(end);\
  cudaEventSynchronize(end);\
  cudaEventElapsedTime(&time_var, start, end);

#ifdef SERIAL
struct _time_s {
  float total;
};

#define PRINTTIME(time, text_tot) {\
  printf(text_tot, time.total);\
}

#define EMPTYTIME(time) {\
  time.total = 0;\
}
 
#else
struct _time_s {
  float total;
  float memcpy;
  float run;
  float memret;
};

#define PRINTTIME(time, text_tot, text_mec, text_run, text_mer) {\
  printf(text_tot, time.total);\
  printf(text_mec, time.memcpy);\
  printf(text_run, time.run);\
  printf(text_mer, time.memret);\
}

#define EMPTYTIME(time) {\
  time.total = 0;\
  time.memcpy = 0;\
  time.run = 0;\
  time.memret = 0;\
}

#endif

#define time_s struct _time_s

void time_add(time_s* to, const time_s* fr);
void time_sub(time_s* to, const time_s* fr);
void time_mul(time_s* to, const float fr);
void time_div(time_s* to, const float fr);

#endif
