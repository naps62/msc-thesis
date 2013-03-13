/*
 * display.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_DISPLAY_H_
#define _PPM_DISPLAY_H_

#include <beast/gl/async_window.hpp>
#include "utils/config.h"

#include <string>
using std::string;

namespace ppm {

struct Display : public beast::gl::async_window {

	Display(const string name, const uint w, const uint h);
	Display(const Config& config);

	void render();
};

}

#endif // _PPM_DISPLAY_H_
