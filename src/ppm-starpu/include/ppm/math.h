#ifndef _PPM_MATH_H_
#define _PPM_MATH_H_

#include "utils/common.h"
#include <cmath>

namespace ppm { namespace math {

#define RAY_EPSILON 1e-4f

template<class T> __HYBRID__ T clamp(T val, T low, T high);
template<class T> __HYBRID__ T max(T a, T b);
template<class T> __HYBRID__ T min(T a, T b);
template<class T> __HYBRID__ T swap(T& a, T& b);
template<class T> __HYBRID__ T mod(T a, T b);

__HYBRID__ inline float radians(float deg);
__HYBRID__ inline float degrees(float rad);
__HYBRID__ inline float sign(float a);
__HYBRID__ inline int   sign(int a);

__HYBRID__ void concentric_sample_disk(const float u1, const float u2, float *dx, float *dy);

} }

#endif // _PPM_MATH_H_
