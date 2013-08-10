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

#ifndef _LUXRAYS_DATASET_H
#define	_LUXRAYS_DATASET_H

#include <deque>

#include "luxrays/core.h"
#include "luxrays/core/accelerator.h"
#include "luxrays/core/trianglemesh.h"

#include "ppm/geometry/bsphere.h"

namespace luxrays {


class DataSet {
public:
	DataSet();
	~DataSet();

	TriangleMeshID Add(Mesh *mesh);
	void Preprocess();
	bool IsPreprocessed() const { return preprocessed; }

	bool Intersect(const Ray *ray, RayHit *hit) const;

	 TriangleMeshID GetMeshID(const unsigned int index) const { return accel->GetMeshID(index); }
	 TriangleMeshID *GetMeshIDTable() const { return accel->GetMeshIDTable(); }
	 TriangleID GetMeshTriangleID(const unsigned int index) const { return accel->GetMeshTriangleID(index); }
	 TriangleID *GetMeshTriangleIDTable() const { return accel->GetMeshTriangleIDTable(); }

	void SetAcceleratorType(AcceleratorType type) { accelType = type; }
	AcceleratorType GetAcceleratorType() const { return accelType; }
	const Accelerator *GetAccelerator() const { return accel; }


	const BBox &GetBBox() const { return bbox; }
	const BSphere &GetBSphere() const { return bsphere; }

	const ppm::BSphere GetPPMBSphere() const {
		ppm::Point center(bsphere.center.x, bsphere.center.y, bsphere.center.z);
		return ppm::BSphere(center, bsphere.rad);
	}

	unsigned int GetTotalVertexCount() const { return totalVertexCount; }
	unsigned int GetTotalTriangleCount() const { return totalTriangleCount; }

	unsigned int GetDataSetID() const { return dataSetID; }
	bool IsEqual(const DataSet *dataSet) const;

	friend class Context;
	friend class OpenCLIntersectionDevice;

protected:
	void UpdateMeshes();

public:
	unsigned int dataSetID;

	//const Context *context;

	unsigned int totalVertexCount;
	unsigned int totalTriangleCount;
	std::deque<Mesh *> meshes;

	bool preprocessed;

	BBox bbox;
	BSphere bsphere;

	AcceleratorType accelType;
	Accelerator *accel;
};

}

#endif	/* _LUXRAYS_DATASET_H */

