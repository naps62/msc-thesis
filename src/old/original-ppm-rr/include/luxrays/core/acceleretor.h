/***************************************************************************
 *   Copyright (C) 1998-2010 by authors (see AUTHORS.txt )                 *
 *                                                                         *
 *   This file is part of LuxRays.                                         *
 *                                                                         *
 *   LuxRays is free software; you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   LuxRays is distributed in the hope that it will be useful,            *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   LuxRays website: http://www.luxrender.net                             *
 ***************************************************************************/

#ifndef _LUXRAYS_ACCELERETOR_H
#define	_LUXRAYS_ACCELERETOR_H

#include "core.h"
#include "luxrays/core/trianglemesh.h"

typedef enum {
	ACCEL_BVH, ACCEL_QBVH, ACCEL_MQBVH
} AcceleratorType;

class Accelerator {
public:
	Accelerator() {
	}
	virtual ~Accelerator() {
	}

	virtual AcceleratorType GetType() const = 0;

	virtual void Init(const std::deque<Mesh *> &meshes,
			const unsigned int totalVertexCount,
			const unsigned int totalTriangleCount) = 0;
	virtual TriangleMeshID GetMeshID(const unsigned int index) const = 0;
	virtual TriangleMeshID *GetMeshIDTable() const = 0;
	virtual TriangleID GetMeshTriangleID(const unsigned int index) const = 0;
	virtual TriangleID *GetMeshTriangleIDTable() const = 0;
	virtual void* GetNodes() const =0;
	virtual uint GetNodesCount()const  =0;
	virtual void* GetPrims()const  =0;
	virtual uint GetPrimsCount() const =0;

	virtual bool Intersect(const Ray *ray, RayHit *hit) const = 0;

};

#endif	/* _LUXRAYS_ACCELERETOR_H */
