#ifndef _PPM_DISPLAY_H_
#define _PPM_DISPLAY_H_

#include <beast/gl/async_window.hpp>
#include "utils/config.h"
#include "ppm/film.h"

namespace ppm {

struct Display : public beast::gl::async_window {

	Display(const Config& config, Film& film);
	void render();

private:
	Film& film;
};

}

#endif // _PPM_DISPLAY_H_
