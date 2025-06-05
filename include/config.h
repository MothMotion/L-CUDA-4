#pragma once
#ifndef CONFIG_H
#define CONFIG_H



#include <stdint.h>



#define ARRAY_SIZE (uint32_t)atoi(getenv("ARRAY_SIZE"))
#define CYCLES (uint32_t)atoi(getenv("CYCLES"))
#define _KBLOCKS (uint32_t)atoi(getenv("KBLOCKS"))
#define KTHREADS (uint32_t)atoi(getenv("KTHREADS"))
#define KBLOCKS ( (_KBLOCKS) ? (_KBLOCKS) : ( (ARRAY_SIZE%KTHREADS) ? (ARRAY_SIZE/KTHREADS + 1) : (ARRAY_SIZE/KTHREADS) ) )

//#define SERIAL

#define arr_t float

#define MIN_RAND 1
#define MAX_RAND 10000000

#endif // !CONFIG_H
