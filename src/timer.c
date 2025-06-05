#include "config.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


void time_add(time_s* to, const time_s* fr) {
  #ifdef SERIAL
  to->total += fr->total;

  #else
  to->total += fr->total;
  to->memcpy += fr->memcpy;
  to->run += fr->run;
  to->memret += fr->memret;

  #endif
}

void time_sub(time_s* to, const time_s* fr) {
  #ifdef SERIAL
  to->total -= fr->total;

  #else
  to->total -= fr->total;
  to->memcpy -= fr->memcpy;
  to->run -= fr->run;
  to->memret -= fr->memret;

  #endif
}

void time_mul(time_s* to, const float fr) {
  #ifdef SERIAL
  to->total *= fr;

  #else
  to->total *= fr;
  to->memcpy *= fr;
  to->run *= fr;
  to->memret *= fr;

  #endif
}

void time_div(time_s* to, const float fr) {
  #ifdef SERIAL
  to->total /= fr;

  #else
  to->total /= fr;
  to->memcpy /= fr;
  to->run /= fr;
  to->memret /= fr;

  #endif
}

#ifdef __cplusplus
}
#endif
