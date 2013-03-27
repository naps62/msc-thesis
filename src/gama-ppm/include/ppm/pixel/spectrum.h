/*
 * spectrum.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_SPECTRUM_H_
#define _PPM_GEOMETRY_SPECTRUM_H_

#include <gamalib/gamalib.h>
#include "ppm/geometry/point.h"
#include "ppm/math.h"

namespace ppm {

struct Spectrum {
	float r, g, b;

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ Spectrum(float _r = 0.f, float _g = 0.f, float _b = 0.f)
	: r(_r), g(_g), b(_b) { }

	// copy constructor
	__HYBRID__ Spectrum(const Spectrum& s)
	: r(s.r), g(s.g), b(s.b) { }

	// copy from luxrays constructor
	Spectrum(const luxrays::Spectrum& s)
	: r(s.r), g(s.g), b(s.b) { }

	// constructor from an array
	__HYBRID__ Spectrum(float s[3])
	: r(s[0]), g(s[1]), b(s[2]) { }

	/*
	 * operators
	 */
	__HYBRID__ Normal operator-() const {
		return Normal(-r, -g, -b);
	}

	__HYBRID__ Spectrum operator+ (const Spectrum& s) const {
		return Spectrum(r + s.r, g + s.g, b + s.b);
	}

	__HYBRID__ Spectrum operator- (const Spectrum& s) const {
		return Spectrum(r - s.r, g - s.g, b - s.b);
	}

	__HYBRID__ Spectrum& operator += (const Spectrum& s) {
		r += s.r; g += s.g; b += s.b;
		return *this;
	}

	__HYBRID__ Spectrum& operator -= (const Spectrum& s) {
		r -= s.r; g -= s.g; b -= s.b;
		return *this;
	}

	__HYBRID__ Spectrum operator* (const float f) const {
		return Spectrum(f * r, f * g, f * b);
	}

	__HYBRID__ Spectrum& operator*= (const float f) {
		r *= f; g *= f; b *= f;
		return *this;
	}

	__HYBRID__ Spectrum operator/ (const float f) const {
		float inv = 1.f / f;
		return Spectrum(inv * r, inv * g, inv * b);
	}

	__HYBRID__ Spectrum& operator/= (const float f) {
		float inv = 1.f / f;
		r *= inv; g *= inv; b *= inv;
		return *this;
	}

	__HYBRID__ bool operator== (const Spectrum& s) {
		return r == s.r && g == s.g && b == s.b;
	}

	__HYBRID__ float operator[](const int i) const {
		return (&r)[i];
	}

	__HYBRID__ float& operator[](const int i) {
		return (&r)[i];
	}

	__HYBRID__ float filter() const {
		return max(r, max(g, b));
	}

	__HYBRID__ bool black() const {
		return r == 0.f && g == 0.f && b == 0.f;
	}

	__HYBRID__ bool is_nan() const {
		return isnan((double)r) || isnan((double)g) || isnan((double)b);
	}

	__HYBRID__ float y() const {
		return 0.212671f * r + 0.715160f * g + 0.072169f * b;
	}

	__HYBRID__ void sclamp() {
		clamp(r, 0.f, 1.f);
		clamp(g, 0.f, 1.f);
		clamp(b, 0.f, 1.f);
	}

	__HYBRID__ Spectrum exp(const Spectrum& s) {
		return Spectrum(expf(s.r), expf(s.g), expf(s.b));
	}
};

ostream& operator<<(ostream& os, const Spectrum& s);

__HYBRID__ __forceinline Spectrum operator*(const float f, const Spectrum &s) {
	return s * f;
}


}

#endif // _PPM_GEOMETRY_SPECTRUM_H_
