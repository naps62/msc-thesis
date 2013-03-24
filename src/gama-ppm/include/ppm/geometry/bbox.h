/*
 * bbox.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_BBOX_H_
#define _PPM_GEOMETRY_BBOX_H_

#include <gama.h>
#include "ppm/geometry/vector.h"
#include "ppm/geometry/point.h"
#include "ppm/geometry/ray.h"
#include "ppm/geometry/bsphere.h"
#include "ppm/math.h"

namespace ppm {

struct BBox {
	Point pmin, pmax;

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ BBox()
	: pmin(Point( INFINITY,  INFINITY,  INFINITY)),
	  pmax(Point(-INFINITY, -INFINITY, -INFINITY))
	{ }

	// constructor from a point
	__HYBRID__ BBox(const Point& p)
	: pmin(p), pmax(p)
	{ }

	// value constructor
	__HYBRID__ BBox(const Point& p1, const Point& p2) {
		pmin = Point(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
		pmax = Point(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
	}

	__HYBRID__ bool overlaps(const BBox& b) const {
		bool x = (pmax.x >= b.pmin.x) && (pmin.x <= b.pmax.x);
		bool y = (pmax.y >= b.pmin.y) && (pmin.y <= b.pmax.y);
		bool z = (pmax.z >= b.pmin.z) && (pmin.z <= b.pmax.z);
		return (x && y && z);
	}

	__HYBRID__ bool inside(const Point& pt) const {
		return (pt.x >= pmin.x && pt.x <= pmax.x &&
				pt.y >= pmin.y && pt.x <= pmax.y &&
				pt.z >= pmin.z && pt.z <= pmax.z);
	}

	__HYBRID__ void expand(const float delta) {
		const Vector delta_vec(delta, delta, delta);
		pmin -= delta_vec;
		pmax -= delta_vec;
	}

	__HYBRID__ float volume() const {
		Vector d = pmax - pmin;
		return d.x * d.y * d.z;
	}

	__HYBRID__ float surface_area() const {
		Vector d = pmax - pmin;
		return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
	}

	__HYBRID__ int maximum_extent() const {
		Vector diag = pmax - pmin;
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	}

// // this method is probably not needed, i should only use the one below
//	__HYBRID__ void bounding_sphere(Point *c, float *rad) const {
//
//	}

	__HYBRID__ BSphere bounding_sphere() const {
		const Point c = .5f * (pmin + pmax);
		const float rad = inside(c) ? c.distance(pmax) : 0.f;
		return BSphere(c, rad);
	}

	__HYBRID__ bool intersect(const Ray& ray, float *hitt0 = NULL, float *hitt1 = NULL) const {
		float t0 = ray.mint;
		float t1 = ray.maxt;
		for(int i = 0; i < 3; ++i) {
			// update interval for _i_th bounding box slab
			float inv_ray_dir = 1.f / ray.d[i];
			float t_near = (pmin[i] - ray.o[i]) * inv_ray_dir;
			float t_far  = (pmax[i] - ray.o[i]) * inv_ray_dir;
			// update parametric interval from slab intersection $t$s
			if (t_near > t_far) ppm::math::swap(t_near, t_far);
			t0 = t_near > t0 ? t_near : t0;
			t1 = t_far  < t1 ? t_far  : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	__HYBRID__ BBox join(const Point& p) const {
		BBox ret;
		ret.pmin.x = ppm::math::min(pmin.x, p.x);
		ret.pmin.y = ppm::math::min(pmin.y, p.y);
		ret.pmin.z = ppm::math::min(pmin.z, p.z);
		ret.pmax.x = ppm::math::max(pmax.x, p.x);
		ret.pmax.y = ppm::math::max(pmax.y, p.y);
		ret.pmax.z = ppm::math::max(pmax.z, p.z);
		return ret;
	}

	__HYBRID__ BBox join(const BBox& b) const {
		BBox ret;
		ret.pmin.x = ppm::math::min(pmin.x, b.pmin.x);
		ret.pmin.y = ppm::math::min(pmin.y, b.pmin.y);
		ret.pmin.z = ppm::math::min(pmin.z, b.pmin.z);
		ret.pmax.x = ppm::math::max(pmax.x, b.pmax.x);
		ret.pmax.y = ppm::math::max(pmax.y, b.pmax.y);
		ret.pmax.z = ppm::math::max(pmax.z, b.pmax.z);
		return ret;
	}

	friend std::ostream& operator<< (std::ostream& os, const BBox& b);

};

ostream& operator<<(ostream& os, const BBox& bbox);



}

#endif // _PPM_GEOMETRY_BBOX_H_
