#include "ppm/engines/ppm.h"

namespace ppm {

PPM :: PPM(const Config& config)
: Engine(config) { }

void PPM :: render() {
	float r = 0.f;
	while(true) {
		film.clear(Spectrum(r, 0.f, 0.f));
		r += 0.01;
		if (r >= 1.f) r = 0.f;
		cout << "r: " << r << endl;
		display->set_require_update();
	}
}

}
