/*
 * update.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef UPDATE_H_
#define UPDATE_H_

#include <gama.h>
#include "GAMAFVMesh2D.h"

class Update : public work {

	FVL::GAMAFVMesh2D mesh;
	smartPtr<double> flux;
	smartPtr<double> polution;

	unsigned start_cell;
	unsigned end_cell;

	__HYBRID__ Update(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _flux, smartPtr<double> _polution)
	: mesh(_mesh), flux(_flux), polution(_polution)
	{
		WORK_TYPE_ID = WORK_UPDATE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ Update(FVL::GAMAFVMesh2D& _mesh, smartPtr<double> _flux, smartPtr<double> _polution, unsigned _start, unsigned _end)
	: mesh(_mesh), flux(_flux), polution(_polution),
	  start_cell(_start), end_cell(_end)
	{
		WORK_TYPE_ID = WORK_COMPUTE_FLUX | W_REGULAR | W_WIDE;
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
			(*L)[k] = new Update(mesh, flux, polution, start, end);
			start = end;
		}

		return L;
	}

	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if (TID > (end_cell - start_cell)) return;

		unsigned long tid = TID + start_cell;
		for(; tid < end_cell; tid += TID_SIZE) {
			//TODO
		}
	}
};

#endif /* UPDATE_H_ */
