#pragma once

#include <gamalib/gamalib.h>
#include <gama_ext/vector.h>

namespace ppm { namespace kernels {

class BuildHitPoints : public work {

	PtrFreeScene* scene;
	uint iteration;
	uint lo, hi;

public:

	__HYBRID__ BuildHitPoints(PtrFreeScene* _scene, uint _iter)
	: scene(_scene), iteration(_iter), lo(0) {
		WORK_TYPE_ID = WORK_BUILD_HIT_POINTS | W_REGULAR | W_WIDE;
	}

};

} }
