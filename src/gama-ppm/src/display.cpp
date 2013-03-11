/*
 * display.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "display.h"
#include <GL/freeglut.h>

//
// constructors
//

// constructor receiving individual parameters
//*************************
Display :: Display(
		const string name,	// name of the window
		const uint w,		// window width
		const uint h)		// window height
: beast::gl::async_window(name, w, h) {
//*************************
}

// constructor receiving a config struct
//*************************
Display :: Display(const Config& config)
: beast::gl::async_window(config.title, config.width, config.height) {
//*************************
}

//
// public methods
//

// render function
//*************************
void Display :: render() {
//*************************
	glMatrixMode(GL_PROJECTION);
	glEnable(GL_SCISSOR_TEST);

	glViewport(0, 0, 400, 300);
	glScissor(0, 0, 400, 300);
	glLoadIdentity();
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(400, 300, 400, 300);
	glScissor(400, 300, 400, 300);
	glLoadIdentity();
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
