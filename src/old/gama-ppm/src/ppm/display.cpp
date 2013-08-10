#include "ppm/display.h"
#include <GL/freeglut.h>

namespace ppm {

//
// constructors
//

Display :: Display(const Config& _config, Film& _film)
: beast::gl::async_window(_config.title, _config.width, _config.height), film(_film), config(_config) {
}

//
// public methods
//

void Display :: render() {
	glRasterPos2i(0, 0);
	// draw frame buffer
	Spectrum* buffer = film.get_frame_buffer_ptr();
	glDrawPixels(film.width, film.height, GL_RGB, GL_FLOAT, buffer);

	// draw caption background
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0.f, 0.f, 0.f, 0.5f);
	glRecti(0, 0, config.width - 1, 18);
	glRecti(0, config.height - 15, config.width - 1, config.height - 1);
	glDisable(GL_BLEND);

//	// header
	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(4, config.height - 11);
	print_string(GLUT_BITMAP_8_BY_13, header);
//
//	// footer
	glRasterPos2i(4, 5);
	print_string(GLUT_BITMAP_8_BY_13, footer);
//	char captionBuffer[512];
//	const double elapsedTime = WallClockTime() - engine->startTime;
//	const unsigned int kPhotonsSec = engine->getPhotonTracedTotal() / (elapsedTime * 1000.f);
//	sprintf(captionBuffer, "[Photons %.2fM][Avg. photons/sec % 4dK][Elapsed time %dsecs]",
//		float(engine->getPhotonTracedTotal() / 1000000.0), kPhotonsSec, int(elapsedTime));
//	PrintString(GLUT_BITMAP_8_BY_13, captionBuffer);
}

void Display :: print_string(void *font, const string& str) const {
	for (uint i(0); i < str.length(); i++)
		glutBitmapCharacter(font, str[i]);
}

void Display :: set_captions(stringstream& header_ss, stringstream& footer_ss) {
	header = header_ss.str();
	footer = footer_ss.str();
}

}
