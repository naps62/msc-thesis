#ifndef _LUXRAYS_H
#define	_LUXRAYS_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "cuda.h"
#include "cuda_runtime.h"



#define LR_LOG(a) { std::cerr << a << std::endl;}
#define LR_LOG_C(a) { fprintf(stderr,a);fprintf(stderr,"\n");}


template<class T>
std::string to_string(T t, std::ios_base & (*f)(std::ios_base&)) {
	std::ostringstream oss;
	oss << f << t;
	return oss.str();
}

#if defined(__CUDACC__)
#define __HD__ 			__device__
#define __H_D__			__host__ __device__
#define __noinline 		__noinline__
#define __forceinline 	__forceinline__


#else
#define __HD__
#define __H_D__
#define __noinline
#define __forceinline 	__inline__ __attribute__((__always_inline__))

#endif



typedef unsigned char u_char;
typedef unsigned short u_short;
typedef unsigned int u_int;
typedef unsigned long u_long;

#include "luxrays/core/geometry/vector.h"
#include "luxrays/core/geometry/normal.h"
#include "luxrays/core/geometry/uv.h"
#include "luxrays/core/geometry/vector_normal.h"
#include "luxrays/core/geometry/point.h"
#include "luxrays/core/geometry/ray.h"
#include "luxrays/core/geometry/raybuffer.h"
#include "luxrays/core/geometry/bbox.h"
#include "luxrays/core/geometry/triangle.h"
#include "luxrays/core/pixel/samplebuffer.h"
#include "random.h"






#endif	/* _LUXRAYS_H */
