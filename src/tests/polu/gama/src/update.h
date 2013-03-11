/*
 * update.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef UPDATE_H_
#define UPDATE_H_

#include "GAMAFVMesh2D.h"

class Update : public work {

	FVL::GAMAFVMesh2D mesh;
	smartPtr<double> flux;
	smartPtr<double> polution;

	const double delta_t;
	const unsigned start_cell;
	const unsigned end_cell;

public:

	__HYBRID__ Update(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _flux, smartPtr<double> _polution, const double _dt)
	: mesh(_mesh), flux(_flux), polution(_polution),
	  delta_t(_dt), start_cell(0), end_cell(mesh.num_cells)
	{
		WORK_TYPE_ID = WORK_UPDATE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ Update(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _flux, smartPtr<double> _polution, const double _dt, const unsigned _start, const unsigned _end)
	: mesh(_mesh), flux(_flux), polution(_polution),
	  delta_t(_dt), start_cell(_start), end_cell(_end)
	{
		WORK_TYPE_ID = WORK_UPDATE | W_REGULAR | W_WIDE;
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int& number) {
		unsigned range = (end_cell - start_cell);
		unsigned number_of_cells = range / number;

		if (number_of_cells == 0) {
			number_of_cells = 1;
			number = range;
		}

		unsigned start = start_cell;
		unsigned end;

		List<work*>* L = new List<work*>(number);
		for(unsigned k = 0; k < number; ++k) {
			end = start + number_of_cells;
			(*L)[k] = new Update(mesh, flux, polution, delta_t, start, end);
			start = end;
		}

		return L;
	}

	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if (TID > (end_cell - start_cell)) return;

		unsigned long tid = TID + start_cell;
		for(; tid < end_cell; tid += TID_SIZE) {
			double global_var = 0;

			for(unsigned e = 0; e < mesh.cell_edges_count[tid]; ++e) {
				unsigned edge = mesh.cell_edges[e][tid];
				double var = delta_t * flux[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[tid];

				if (mesh.edge_left_cells[edge] == tid) {
					global_var -= var;
				} else {
					global_var += var;
				}
			}
			polution[tid] += global_var;
		}
	}
};

#endif /* UPDATE_H_ */
