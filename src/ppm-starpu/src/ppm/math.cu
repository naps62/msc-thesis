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

__HYBRID__
void concentric_sample_disk(const float u1, const float u2, float *dx, float *dy) {
  float r, theta;
  // Map uniform random numbers to $[-1,1]^2$
  float sx = 2.f * u1 - 1.f;
  float sy = 2.f * u2 - 1.f;
  // Map square to $(r,\theta)$
  // Handle degeneracy at the origin
  if (sx == 0.f && sy == 0.f) {
    *dx = 0.f;
    *dy = 0.f;
    return;
  }
  if (sx >= -sy) {
    if (sx > sy) {
      // Handle first region of disk
      r = sx;
      if (sy > 0.f)
        theta = sy / r;
      else
        theta = 8.f + sy / r;
    } else {
      // Handle second region of disk
      r = sy;
      theta = 2.f - sx / r;
    }
  } else {
    if (sx <= sy) {
      // Handle third region of disk
      r = -sx;
      theta = 4.f - sy / r;
    } else {
      // Handle fourth region of disk
      r = -sy;
      theta = 6.f + sx / r;
    }
  }
  theta *= M_PI / 4.f;
  *dx = r * cosf(theta);
  *dy = r * sinf(theta);
}

} }
