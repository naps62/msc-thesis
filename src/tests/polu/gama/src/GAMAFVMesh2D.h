/*
 * GAMAFVMesh2D.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef GAMAFVMESH2D_H_
#define GAMAFVMESH2D_H_

#include "FVL/FVMesh2D_SOA.h"
#include <gamalib/memlib/smartpointer.h>

namespace FVL {

	struct GAMAFVMesh2D {

		// Vertex info
		unsigned num_vertex;

		// Edge info
		unsigned num_edges;
		smartPtr<double>   edge_lengths;
		smartPtr<unsigned> edge_left_cells;
		smartPtr<unsigned> edge_right_cells;
		smartPtr<double>   edge_normals_x;
		smartPtr<double>   edge_normals_y;

		// Cell info
		unsigned num_cells;
		smartPtr<double>   cell_areas;
		smartPtr<unsigned> cell_edges_count;
		smartPtr<unsigned> cell_edges[3];

		GAMAFVMesh2D();
		GAMAFVMesh2D(const string& filename);

	private:

		void import_FVMesh2D(FVMesh2D &msh);

		void alloc();
	};
}


#endif /* GAMAFVMESH2D_H_ */
