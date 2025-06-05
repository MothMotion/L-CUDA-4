#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "config.h"

#include <stdint.h>

#define MAX(max, value) \
  ( (max > value) ? (max) : (value) )

#define MIN(min, value) \
  ( (min < value) ? (min) : (value) )

void Randomize(arr_t* arr, const uint32_t size, const arr_t min_v, const arr_t max_v);

#endif // !RANDOM_H
