/*
 * display.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef DISPLAY_H_
#define DISPLAY_H_

#include <beast/gl/async_window.hpp>
#include "config.h"

#include <string>
using std::string;

struct Display : public beast::gl::async_window {

	Display(const string name, const uint w, const uint h);
	Display(const Config& config);

	void render();
};

#endif // DISPLAY_H_
