#include "ppm/display.h"
#include <GL/freeglut.h>

namespace ppm {

//
// constructors
//

Display :: Display(const Config& config, Film& _film)
: beast::gl::async_window(config.title, config.width, config.height), film(_film) {
}

//
// public methods
//

void Display :: render() {
	Spectrum* buffer = film.get_frame_buffer_ptr();
	glDrawPixels(film.width, film.height, GL_RGB, GL_FLOAT, buffer);
}

void Display :: set_require_update() {
	glutPostRedisplay();
}

}
