#ifdef SERIAL


#include "config.h"
#include "matrix.h"
#include "timer.h"
#include "vec_oper.h"

#include <stdint.h>
#include <stddef.h>



time_s Operation(const matrix& mat1, const matrix& mat2, matrix& out, const Oper& op) {
  time_s (*func)(arr_t*, arr_t*, arr_t*, const uint32_t&) = NULL;
  time_s result;
  switch(op) {
    case opadd : func = Sum; break;
    case opsub : func = Sub; break;
    case opmul : func = Mul; break;
    case opdiv : func = Div; break;
    default : break;
  }

  for(uint32_t i=0; i<out.size; ++i)
    result += func(mat1.data[i], mat2.data[i], out.data[i], out.size) * 1000;

  return result;
}



time_s Sum(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  time_s time;
  GETTIME({
    for(uint32_t i=0; i<size; ++i)
      out[i] = arr1[i] + arr2[i];
  }, time.total);

  return time;
}

time_s Sub(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  time_s time;
  GETTIME({
    for(uint32_t i=0; i<size; ++i)
      out[i] = arr1[i] - arr2[i];
  }, time.total);

  return time;
}

time_s Mul(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  time_s time;
  GETTIME({
    for(uint32_t i=0; i<size; ++i)
      out[i] = arr1[i] * arr2[i];
  }, time.total);

  return time;
}

time_s Div(arr_t* arr1, arr_t* arr2, arr_t* out, const uint32_t& size) {
  time_s time;
  GETTIME({
    for(uint32_t i=0; i<size; ++i)
      if(arr2[i])
        out[i] = arr1[i] / arr2[i];
      else out[i] = arr1[i];
  }, time.total);

  return time;
}

#endif
