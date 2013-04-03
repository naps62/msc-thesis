#ifndef _PPM_ENGINE_PPM_H_
#define _PPM_ENGINE_PPM_H_

#include "ppm/engine.h"

namespace ppm {

class PPM : public Engine {

public:
	PPM(const Config& _config);
	void render();
	void set_captions();
};

}

#endif // _PPM_ENGINE_PPM_H_
