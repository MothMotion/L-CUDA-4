#pragma once
#ifndef CONFIG_H
#define CONFIG_H



#include <stdint.h>



#define ARRAY_SIZE (uint32_t)atoi(getenv("ARRAY_SIZE"))
#define CYCLES (uint32_t)atoi(getenv("CYCLES"))
#define KBLOCKS (uint32_t)atoi(getenv("KBLOCKS"))
#define KTHREADS (uint32_t)atoi(getenv("KTHREADS"))

//#define SERIAL

#define arr_t float

#define MIN_RAND 1
#define MAX_RAND 10000000

#endif // !CONFIG_H
