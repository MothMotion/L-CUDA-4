#pragma once
#ifndef MATRIX_H
#define MATRIX_H



#include "config.h"



struct matrix {
  arr_t** data = nullptr;
  uint32_t size = 0;

  // data must be nullptr
  void init();
  void deinit();

  // dangerous, free after use
  void flat(arr_t* to) const;
  // caution, frees 'from'
  void deflat(arr_t* from);
};

#endif // !MATRIX_H
