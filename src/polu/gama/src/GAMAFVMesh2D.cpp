#include "GAMAFVMesh2D.h"

namespace FVL {


	GAMAFVMesh2D::GAMAFVMesh2D(const string& filename) {
		FVMesh2D mesh(filename);
		import_FVMesh2D(mesh);
	}


	void GAMAFVMesh2D::import_FVMesh2D(FVMesh2D& msh) {
		num_vertex	= msh.getNbVertex();
		num_edges	= msh.getNbEdge();
		num_cells	= msh.getNbCell();

		// allocs space for all needed data
		alloc();

		// copy edge data
		FVEdge2D *edge;
		int i = 0;
		for(msh.beginEdge(); (edge = msh.nextEdge()); ++i) {
			// edge normal
			edge_normals_x[i]	= edge->normal.x;
			edge_normals_y[i]	= edge->normal.y;

			// edge length
			edge_lengths[i] 	= edge->length;

			// edge left cell (always exists)
			edge_left_cells[i]	= edge->leftCell->label - 1;

			// edge right cell (need check. if border edge, rightCell is null)
			edge_right_cells[i]	= (edge->rightCell != NULL) ? (edge->rightCell->label - 1) : NO_RIGHT_CELL;
		}

		// copy cell data
		FVCell2D *cell;
		i = 0;
		//num_total_edges = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
			// cell area
			cell_areas[i]	= cell->area;

			// count of edges for this cell
			cell_edges_count[i] = cell->nb_edge;
		}
	}

	void GAMAFVMesh2D::alloc() {
		if (num_vertex <= 0 || num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		// alloc edge info
		edge_lengths		= smartPtr<double>  (sizeof(double) * num_edges);
		edge_left_cells		= smartPtr<unsigned>(sizeof(unsigned) * num_edges);
		edge_right_cells	= smartPtr<unsigned>(sizeof(unsigned) * num_edges);
		edge_normals_x		= smartPtr<double>  (sizeof(double) * num_edges);
		edge_normals_y		= smartPtr<double>  (sizeof(double) * num_edges);


		// alloc cell info
		cell_areas			= smartPtr<double>(sizeof(double) * num_cells);
		cell_edges_count	= smartPtr<unsigned>(sizeof(unsigned) * num_cells);
		for(unsigned int i = 0; i < 3; ++i) {
			cell_edges[i] = smartPtr<unsigned>(sizeof(unsigned) * num_cells);
		}
	}

	}
}

