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

#ifndef _LUXRAYS_MQBVHACCEL_H
#define	_LUXRAYS_MQBVHACCEL_H

#include <string.h>
#include <xmmintrin.h>
#include <boost/cstdint.hpp>

#include "core.h"
#include "luxrays/core/acceleretor.h"
#include "luxrays/accelerators/qbvhaccel.h"

using boost::int32_t;


class MQBVHAccel  : public Accelerator {
public:
	MQBVHAccel(const Context *context, u_int fst, u_int sf);
	~MQBVHAccel();

	BBox WorldBound() const;

	AcceleratorType GetType() const { return ACCEL_QBVH; }
	void Init(const std::deque<Mesh *> &meshes, const unsigned int totalVertexCount,
		const unsigned int totalTriangleCount);

	const TriangleMeshID GetMeshID(const unsigned int index) const { return meshIDs[index]; }
	const TriangleMeshID *GetMeshIDTable() const { return meshIDs; }
	const TriangleID GetMeshTriangleID(const unsigned int index) const { return meshTriangleIDs[index]; }
	const TriangleID *GetMeshTriangleIDTable() const { return meshTriangleIDs; }

	bool Intersect(const Ray *ray, RayHit *hit) const;

	void Update();

	friend class OpenCLIntersectionDevice;

private:
	static bool MeshPtrCompare(Mesh *, Mesh *);

	void BuildTree(u_int start, u_int end, u_int *primsIndexes,
		BBox *primsBboxes, Point *primsCentroids, const BBox &nodeBbox,
		const BBox &centroidsBbox, int32_t parentIndex, int32_t childIndex,
		int depth);

	void CreateLeaf(int32_t parentIndex, int32_t childIndex,
		u_int start, const BBox &nodeBbox);

	int32_t CreateNode(int32_t parentIndex, int32_t childIndex, const BBox &nodeBbox);

	std::deque<Mesh *> meshList;

	QBVHNode *nodes;
	u_int nNodes, maxNodes;
	BBox worldBound;

	u_int fullSweepThreshold;
	u_int skipFactor;

	u_int nLeafs;
	std::map<Mesh *, QBVHAccel *, bool (*)(Mesh *, Mesh *)> accels;
	QBVHAccel **leafs;
	const Transform **leafsInvTransform;
	unsigned int *leafsOffset;
	TriangleMeshID *meshIDs;
	TriangleID *meshTriangleIDs;

	const Context *ctx;
	bool initialized;
};



#endif	/* _LUXRAYS_MQBVHACCEL_H */
