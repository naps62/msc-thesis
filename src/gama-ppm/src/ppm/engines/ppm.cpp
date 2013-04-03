#include "ppm/engines/ppm.h"

namespace ppm {

PPM :: PPM(const Config& config)
: Engine(config) { }

void PPM :: render() {
	float r = 0.f;
	float frame_time = 1.f / config.max_refresh_rate;
	cout << frame_time << endl;
	bool x = true;
	while(true) {
		film.clear(Spectrum(r, 0.f, 0.f));
		r += 0.01;
		if (r >= 1.f) r = 0.f;
		cout << endl;
		display->request_update(frame_time);
	}
}

}
