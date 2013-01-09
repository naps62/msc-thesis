/*
 * compute_flux.cuh
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef COMPUTE_FLUX_H_
#define COMPUTE_FLUX_H_

#include "FVL/FVLib.h"
#include "GAMAFVMesh2D.h"

class ComputeFlux : public work {

	FVL::GAMAFVMesh2D mesh;
	smartPtr<double> velocity;
	smartPtr<double> flux;
	smartPtr<double> polution;

	const unsigned dc;
	const unsigned start_edge;
	const unsigned end_edge;

public:

	__HYBRID__ ComputeFlux(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _velocity, smartPtr<double> _flux, smartPtr<double> _polution, const unsigned _dc)
	: mesh(_mesh), velocity(_velocity), flux(_flux), polution(_polution),
	  dc(_dc), start_edge(0), end_edge(mesh.num_edges)
	{
		WORK_TYPE_ID = WORK_COMPUTE_FLUX | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ComputeFlux(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _velocity, smartPtr<double> _flux, smartPtr<double> _polution, const unsigned _dc, const unsigned _start, const unsigned _end)
	: mesh(_mesh), velocity(_velocity), flux(_flux), polution(_polution),
	  dc(_dc), start_edge(_start), end_edge(_end)
	{
		WORK_TYPE_ID = WORK_COMPUTE_FLUX | W_REGULAR | W_WIDE;
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {
		unsigned range = (end_edge - start_edge);
		unsigned number_of_edges = range / number;

		if (number_of_edges == 0) {
			number_of_edges = 1;
			number = range;
		}

		unsigned start = start_edge;
		unsigned end;

		List<work*>* L = new List<work*>(number);
		for(unsigned k = 0; k < number; ++k) {
			end = start + number_of_edges;
			(*L)[k] = new ComputeFlux(mesh, velocity, flux, polution, dc, start, end);
			start = end;
		}

		return L;
	}

	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if (TID > (end_edge - start_edge)) return;

		for(unsigned long tid = TID + start_edge; tid < end_edge; tid += TID_SIZE) {

			double polution_left = polution[mesh.edge_left_cells[tid]];
			double polution_right =
					(mesh.edge_right_cells[tid] == NO_RIGHT_CELL)
					? dc
					: polution[mesh.edge_right_cells[tid]];

			flux[tid] = (velocity[tid] < 0)
					? velocity[tid] * polution_left
					: velocity[tid] * polution_right;
		}
	}
};

#endif /* COMPUTE_FLUX_H_ */
