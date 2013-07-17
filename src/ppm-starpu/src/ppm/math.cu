#include "ppm/math.h"

namespace ppm { namespace math {

template<class T> __HYBRID__ T clamp(T val, T low, T high) {
  return val > low ? (val < high ? val : high) : low;
}

template<class T> __HYBRID__ T max(T a, T b) {
  return a > b ? a : b;
}

template<class T> __HYBRID__ T min(T a, T b) {
  return a < b ? a : b;
}

template<class T> __HYBRID__ T swap(T& a, T& b) {
  const T tmp = a;
  a = b;
  b = tmp;
}

template<class T> __HYBRID__ T mod(T a, T b) {
  if (b == 0)
    b = 1;
  a %= b;
  if (a < 0)
    a += b;
  return a;
}

__HYBRID__ inline float radians(float deg) {
  return (M_PI / 180.f) * deg;
}

__HYBRID__ inline float degrees(float rad) {
  return (180.f / M_PI) * rad;
}

__HYBRID__ inline float sign(float a) {
  return a < 0.f ? -1.f : 1.f;
}

__HYBRID__ inline int sign(int a) {
  return a < 0 ? -1 : 1;
}

} }
