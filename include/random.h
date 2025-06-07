#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "config.h"
#include "matrix.h"

#include <stdint.h>



void Randomize(const matrix& mat, const arr_t& min_v, const arr_t& max_v);

#endif // !RANDOM_H
