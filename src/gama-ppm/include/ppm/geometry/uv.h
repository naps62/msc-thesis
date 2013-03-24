/*
 * uv.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_UV_H_
#define _PPM_GEOMETRY_UV_H_

#include <gama.h>

namespace ppm {

struct UV {
	float u, v;

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ UV(float _u = 0.f, float _v = 0.f)
	: u(_u), v(_v) { }

	// copy constructor
	__HYBRID__ UV(const UV& uv)
	: u(uv.u), v(uv.v) { }

	// copy from luxrays constructor
	UV(const luxrays::UV& uv)
	: u(uv.u), v(uv.v) { }

	// constructor from an array
	__HYBRID__ UV(float uv[2])
	: u(uv[0]), v(uv[1]) { }

	/*
	 * operators
	 */
	__HYBRID__ UV operator+ (const UV& uv) const {
		return UV(u + uv.u, v + uv.v);
	}

	__HYBRID__ UV operator- (const UV& uv) const {
		return UV(u - uv.u, v - uv.v);
	}

	__HYBRID__ UV& operator += (const UV& uv) {
		u += uv.u; v += uv.v;
		return *this;
	}

	__HYBRID__ UV& operator -= (const UV& uv) {
		u -= uv.u; v -= uv.v;
		return *this;
	}

	__HYBRID__ UV operator* (const float f) const {
		return UV(f * u, f * v);
	}

	__HYBRID__ UV& operator*= (const float f) {
		u *= f; v *= f;
		return *this;
	}

	__HYBRID__ UV operator/ (const float f) const {
		float inv = 1.f / f;
		return UV(inv * u, inv * v);
	}

	__HYBRID__ UV& operator/= (const float f) {
		float inv = 1.f / f;
		u *= inv; v *= inv;
		return *this;
	}

	__HYBRID__ float operator[](const int i) const {
		return (&u)[i];
	}

	__HYBRID__ float& operator[](const int i) {
		return (&u)[i];
	}
};

ostream& operator<<(ostream& os, const UV& uv);

__HYBRID__ __forceinline UV operator*(const float f, const UV &uv) {
	return uv * f;
}

}

#endif // _PPM_GEOMETRY_UV_H_
