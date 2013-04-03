#include "ppm/engines/ppm.h"

namespace ppm {

PPM :: PPM(const Config& config)
: Engine(config) { }

void PPM :: render() {
	float r = 0.f;
	float frame_time = 1.f / config.max_refresh_rate;
	bool x = true;
	while(true) {
		film.clear(Spectrum(r, 0.f, 0.f));
		r += 0.01;
		if (r >= 1.f) r = 0.f;
		set_captions();
		display->request_update(frame_time);
	}
}

void PPM :: set_captions() {
	stringstream header, footer;
	header << "Hello World!";
	footer << "[Photons " << 0 << "M][Avg. photons/sec " << 0 << "K][Elapsed time " << 0 << "secs]";
	display->set_captions(header, footer);
}

}
