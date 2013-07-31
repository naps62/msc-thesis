/*
 * hitpoints.h
 *
 *  Created on: Jul 26, 2012
 *      Author: rr
 */

#ifndef HITPOINTS_H_
#define HITPOINTS_H_

#include "core.h"
#include "pointerfreescene.h"
#include "random.h"
#include "config.h"

class EyePath {
public:
	// Screen information
	float scrX, scrY;

	// Eye path information
	Ray ray;
	unsigned int depth;
	Spectrum throughput;

	bool done;
	bool splat;
	uint sampleIndex;
};

class PhotonPath {
public:
	// The ray is stored in the RayBuffer and the index is implicitly stored
	// in the array of PhotonPath
	Spectrum flux;
	unsigned int depth;
	bool done;
	//Seed seed;
};

//------------------------------------------------------------------------------
// Eye path hit points
//------------------------------------------------------------------------------

enum HitPointType {
	SURFACE, CONSTANT_COLOR
};

//class IterationHitPoint {
//public:
//	//float photonRadius2;
//	unsigned int accumPhotonCount;
//	Spectrum accumReflectedFlux;
//	Spectrum reflectedFlux;
//	//unsigned int photonCount;
//
//	Spectrum accumRadiance;
//	unsigned int constantHitsCount;
//	unsigned long long photonCount; // used only in dependent radius redcution
//
//	float accumPhotonRadius2;
//	unsigned int surfaceHitsCount;
//	Spectrum radiance;
//
//	HitPointType type;
//
//	float scrX, scrY;
//
//	// Used for CONSTANT_COLOR and SURFACE type
//	Spectrum throughput;
//
//	// Used for SURFACE type
//	Point position;
//	Vector wo;
//	Normal normal;
//	//const SurfaceMaterial *material;
//
//	uint materialSS;
//
//	float initialRadius;

//};

/**
 * static information of the hitpoint
 */
class HitPoint {
public:

	uint id;

	HitPointType type;

	float scrX, scrY;

	// Used for CONSTANT_COLOR and SURFACE type
	Spectrum throughput;

	// Used for SURFACE type
	Point position;
	Vector wo;
	Normal normal;

	uint materialSS;

	uint accumPhotonCount;
	Spectrum accumReflectedFlux;
	Spectrum radiance;
	uint constantHitsCount;
	uint surfaceHitsCount;
	Spectrum accumRadiance;
	uint photonCount;
	Spectrum reflectedFlux;

	float accumPhotonRadius2;

	__H_D__
	Point GetPosition() {
		return position;
	}

	__H_D__
	float GetPosition(int p) {
		return position[p];
	}

	__H_D__
	HitPointType GetType() {
		return type;
	}

};

typedef struct s_photonHit {
	Point hitPoint;
	Normal shadeN;
	Vector wi;
	Spectrum photonFlux;
	__H_D__
	Point GetPosition() {
		return hitPoint;
	}

	__H_D__
	float GetPosition(int p) {
		return hitPoint[p];
	}

	/**
	 * hack to make it compitable with
	 * hitpoints
	 */
	__H_D__
	HitPointType GetType() {
		return SURFACE;
	}

} PhotonHit;

__HD__
inline unsigned int Hash(const int ix, const int iy, const int iz,
		unsigned int hashGridSize) {
	return (unsigned int) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791))
			% hashGridSize;
}

#endif /* HITPOINTS_H_ */
