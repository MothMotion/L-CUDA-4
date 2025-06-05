#pragma once
#if !defined(K_VEC_WRAP_H) && !defined(SERIAL)
#define K_VEC_WRAP_H



#include "config.h"
#include "vec_oper.h"
#include "time.h"



extern time_s Operation(arr_t** arr_inp1, arr_t** arr_inp2, arr_t** arr_out, const uint32_t size, const enum Oper operation);



#endif // !K_VEC_WRAP_H
