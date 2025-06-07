#pragma once
#ifndef CONFIG_H
#define CONFIG_H



#include <stdint.h>
#include <stdlib.h>



const uint32_t ARRAY_SIZE = (uint32_t)atoi(getenv("ARRAY_SIZE"));
const uint32_t CYCLES = (uint32_t)atoi(getenv("CYCLES"));
const uint32_t _KBLOCKS = (uint32_t)atoi(getenv("KBLOCKS"));
const uint32_t KTHREADS = (uint32_t)atoi(getenv("KTHREADS"));
const uint32_t KBLOCKS = (_KBLOCKS) ? (_KBLOCKS) :
  ( (ARRAY_SIZE*ARRAY_SIZE%KTHREADS) ? (ARRAY_SIZE*ARRAY_SIZE/KTHREADS + 1) : (ARRAY_SIZE*ARRAY_SIZE/KTHREADS) );

//#define SERIAL

#define arr_t float

const arr_t MIN_RAND = 1;
const arr_t MAX_RAND = 10000000;

#endif // !CONFIG_H
