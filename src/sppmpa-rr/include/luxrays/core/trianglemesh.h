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

#ifndef _LUXRAYS_TRIANGLEMESH_H
#define	_LUXRAYS_TRIANGLEMESH_H

#include <cassert>
#include <cstdlib>

#include "core.h"
#include "luxrays/core/geometry/triangle.h"
#include "geometry/transform.h"


typedef unsigned int TriangleMeshID;
typedef unsigned int TriangleID;

enum MeshType {
	MESH_TYPE_TRIANGLE, TYPE_TRIANGLE_INSTANCE,
	TYPE_EXT_TRIANGLE, TYPE_EXT_TRIANGLE_INSTANCE
};

class Mesh {
public:
	Mesh() { }
	virtual ~Mesh() { }

	virtual MeshType GetType() const = 0;

	virtual unsigned int GetTotalVertexCount() const = 0;
	virtual unsigned int GetTotalTriangleCount() const = 0;

	virtual BBox GetBBox() const = 0;
	__HD__
	virtual Point GetVertex(const unsigned int vertIndex) const = 0;
	__HD__
	virtual float GetTriangleArea(const unsigned int triIndex) const = 0;
	__HD__
	virtual Point *GetVertices() const = 0;
	__HD__
	virtual Triangle *GetTriangles() const = 0;

	virtual void ApplyTransform(const Transform &trans) = 0;
};

class TriangleMesh : public Mesh {
public:
	// NOTE: deleting meshVertices and meshIndices is up to the application
	TriangleMesh(const unsigned int meshVertCount, const unsigned int meshTriCount,
			Point *meshVertices, Triangle *meshTris) {
		assert (meshVertCount > 0);
		assert (meshTriCount > 0);
		assert (meshVertices != NULL);
		assert (meshTris != NULL);

		vertCount = meshVertCount;
		triCount = meshTriCount;
		vertices = meshVertices;
		tris = meshTris;
	};
	virtual ~TriangleMesh() { };
	virtual void Delete() {
		delete[] vertices;
		delete[] tris;
	}

	virtual MeshType GetType() const { return MESH_TYPE_TRIANGLE; }
	unsigned int GetTotalVertexCount() const { return vertCount; }
	unsigned int GetTotalTriangleCount() const { return triCount; }

	BBox GetBBox() const;
	__HD__
	Point GetVertex(const unsigned int vertIndex) const { return vertices[vertIndex]; }
	__HD__
	float GetTriangleArea(const unsigned int triIndex) const { return tris[triIndex].Area(vertices); }
	__HD__
	Point *GetVertices() const { return vertices; }
	__HD__
	Triangle *GetTriangles() const { return tris; }

	virtual void ApplyTransform(const Transform &trans);

	static TriangleMesh *Merge(
		const std::deque<Mesh *> &meshes,
		TriangleMeshID **preprocessedMeshIDs = NULL,
		TriangleID **preprocessedMeshTriangleIDs = NULL);
	static TriangleMesh *Merge(
		const unsigned int totalVerticesCount,
		const unsigned int totalIndicesCount,
		const std::deque<Mesh *> &meshes,
		TriangleMeshID **preprocessedMeshIDs = NULL,
		TriangleID **preprocessedMeshTriangleIDs = NULL);

protected:
	unsigned int vertCount;
	unsigned int triCount;
	Point *vertices;
	Triangle *tris;
};

class InstanceTriangleMesh : public Mesh {
public:
	InstanceTriangleMesh(TriangleMesh *m, const Transform &t) {
		assert (mesh != NULL);

		trans = t;
		invTrans = t.GetInverse();
		mesh = m;
	};
	virtual ~InstanceTriangleMesh() { };

	virtual MeshType GetType() const { return TYPE_TRIANGLE_INSTANCE; }
	unsigned int GetTotalVertexCount() const { return mesh->GetTotalVertexCount(); }
	unsigned int GetTotalTriangleCount() const { return mesh->GetTotalTriangleCount(); }

	BBox GetBBox() const {
		return trans(mesh->GetBBox());
	}
	__HD__
	Point GetVertex(const unsigned int vertIndex) const { return trans(mesh->GetVertex(vertIndex)); }
	__HD__
	float GetTriangleArea(const unsigned int triIndex) const {
		const Triangle &tri = mesh->GetTriangles()[triIndex];

		return Triangle::Area(GetVertex(tri.v[0]), GetVertex(tri.v[1]), GetVertex(tri.v[2]));
	}

	virtual void ApplyTransform(const Transform &t) { trans = trans * t; }

	const Transform &GetTransformation() const { return trans; }
	const Transform &GetInvTransformation() const { return invTrans; }
	__HD__
	Point *GetVertices() const { return mesh->GetVertices(); }
	__HD__
	Triangle *GetTriangles() const { return mesh->GetTriangles(); }
	TriangleMesh *GetTriangleMesh() const { return mesh; };

protected:
	Transform trans, invTrans;
	TriangleMesh *mesh;
};



#endif	/* _LUXRAYS_TRIANGLEMESH_H */
