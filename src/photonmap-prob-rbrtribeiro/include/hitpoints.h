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
class HitPointPositionInfo {
public:

	HitPointType type;

	float scrX, scrY;

	// Used for CONSTANT_COLOR and SURFACE type
	Spectrum throughput;

	// Used for SURFACE type
	Point position;
	Vector wo;
	Normal normal;

	uint materialSS;

};

class HitPointRadianceFlux {
public:

	unsigned int accumPhotonCount;
	Spectrum accumReflectedFlux;
	Spectrum radiance;

	unsigned long long photonCount;
	Spectrum reflectedFlux;

#if defined USE_SPPM || defined USE_PPM
	float accumPhotonRadius2;
#endif


};

__HD__
inline unsigned int Hash(const int ix, const int iy, const int iz, unsigned int hashGridSize) {
	return (unsigned int) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % hashGridSize;
}

#endif /* HITPOINTS_H_ */
