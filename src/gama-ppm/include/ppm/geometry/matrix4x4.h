/*
 * matrix4x4.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_MATRIX4X4_H_
#define _PPM_GEOMETRY_MATRIX4X4_H_

#include <gama.h>
#include <ostream>
using std::ostream;

namespace ppm {

struct Matrix4x4 {
	float m[4][4];

	// empty constructor
	__HYBRID__ Matrix4x4() {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				if (i == j) m[i][j] = 1.f;
				else        m[i][j] = 0.f;
	}

	// constructor from copy
	__HYBRID__ Matrix4x4(const Matrix4x4& mat) {
		set(mat);
	}
	__HYBRID__ Matrix4x4(const luxrays::Matrix4x4& mat) {
		set(mat);
	}

	// constructor from array
	__HYBRID__ Matrix4x4(const float mat[4][4]) {
		set(mat);
	}

	// constructor from values
	Matrix4x4(float t00, float t01, float t02, float t03,
			float t10, float t11, float t12, float t13,
			float t20, float t21, float t22, float t23,
			float t30, float t31, float t32, float t33) {
		m[0][0] = t00; m[0][1] = t01; m[0][2] = t02; m[0][3] = t03;
		m[1][0] = t10; m[1][1] = t11; m[1][2] = t12; m[1][3] = t13;
		m[2][0] = t20; m[2][1] = t21; m[2][2] = t22; m[2][3] = t23;
		m[3][0] = t30; m[3][1] = t31; m[3][2] = t32; m[3][3] = t33;
	}

	__HYBRID__ void set(const Matrix4x4& mat) {
		set(mat.m);
	}

	__HYBRID__ void set(const luxrays::Matrix4x4& mat) {
		set(mat.m);
	}

	__HYBRID__ void set(const float mat[4][4]) {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				m[i][j] = mat[i][j];
	}


	__HYBRID__ Matrix4x4 transpose() const {
		return Matrix4x4(
				m[0][0], m[1][0], m[2][0], m[3][0],
				m[0][1], m[1][1], m[2][1], m[3][1],
				m[0][2], m[1][2], m[2][2], m[3][2],
				m[0][3], m[1][3], m[2][3], m[3][3]);
	}

	__HYBRID__ float determinant() const {
		// row expansion along the last row
		// for most matrices this would be most efficient
		float result = 0;
		float s = -1;
		float A[3][3];

		// initialize for first expansion
		for(int i = 0; i < 3; ++i) {
			for(int j = 0; j < 3; ++j) {
				A[i][j] = m[i][j+1];
			}
		}

		int k = 0;
		while(true) {
			if (m[3][k] != 0.f)
				result += s * m[3][k] * Matrix4x4::det3x3(A);
			// check if we're done
			if (k >= 3)
				break;
			s *= -1;
			// copy column for next expansion
			for(int i = 0; i < 3; ++i)
				A[i][k] = m[i][k];
			++k;
		}
		return result;
	}

	__HYBRID__ Matrix4x4 inverse() const {
		int indxc[4], indxr[4];
		int ipiv[4] = {0, 0, 0, 0};
		float minv[4][4];
		memcpy(minv, m, 4 * 4 * sizeof (float));
		for (int i = 0; i < 4; ++i) {
			int irow = -1, icol = -1;
			float big = 0.;
			// Choose pivot
			for (int j = 0; j < 4; ++j) {
				if (ipiv[j] != 1) {
					for (int k = 0; k < 4; ++k) {
						if (ipiv[k] == 0) {
							if (fabsf(minv[j][k]) >= big) {
								big = fabsf(minv[j][k]);
								irow = j;
								icol = k;
							}
						} else if (ipiv[k] > 1)
							throw std::runtime_error("Singular matrix in MatrixInvert");
					}
				}
			}
			++ipiv[icol];
			// Swap rows _irow_ and _icol_ for pivot
			if (irow != icol) {
				for (int k = 0; k < 4; ++k)
					ppm::math::swap(minv[irow][k], minv[icol][k]);
			}
			indxr[i] = irow;
			indxc[i] = icol;
			if (minv[icol][icol] == 0.)
				throw std::runtime_error("Singular matrix in MatrixInvert");
			// Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
			float pivinv = 1.f / minv[icol][icol];
			minv[icol][icol] = 1.f;
			for (int j = 0; j < 4; ++j)
				minv[icol][j] *= pivinv;
			// Subtract this row from others to zero out their columns
			for (int j = 0; j < 4; ++j) {
				if (j != icol) {
					float save = minv[j][icol];
					minv[j][icol] = 0;
					for (int k = 0; k < 4; ++k)
						minv[j][k] -= minv[icol][k] * save;
				}
			}
		}
		// Swap columns to reflect permutation
		for (int j = 3; j >= 0; --j) {
			if (indxr[j] != indxc[j]) {
				for (int k = 0; k < 4; ++k)
					ppm::math::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
			}
		}
		return Matrix4x4(minv);
	}

	__HYBRID__ static Matrix4x4 mul(const Matrix4x4 &m1, const Matrix4x4 &m2) {
		float r[4][4];
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				r[i][j] = m1.m[i][0] * m2.m[0][j] +
					m1.m[i][1] * m2.m[1][j] +
					m1.m[i][2] * m2.m[2][j] +
					m1.m[i][3] * m2.m[3][j];

		return Matrix4x4(r);
	}

private:

	__HYBRID__ static float det2x2(const float a00, const float a01, const float a10, const float a11) {
		return a00 * a11 - a01 * a10;
	}

	__HYBRID__ static float det3x3(float A[3][3]) {
		return
				A[0][0] * Matrix4x4::det2x2(A[1][1], A[1][2], A[2][1], A[2][2]) -
				A[0][1] * Matrix4x4::det2x2(A[1][0], A[1][2], A[2][0], A[2][2]) +
				A[0][2] * Matrix4x4::det2x2(A[1][0], A[1][1], A[2][0], A[2][1]);
	}
};

ostream & operator<<(ostream &os, const Matrix4x4 &mat);

}

#endif // _PPM_GEOMETRY_MATRIX4X4_H_/