#ifndef _PPM_DISPLAY_H_
#define _PPM_DISPLAY_H_

#include "utils/common.h"
#include <beast/gl/async_window.hpp>
#include "utils/config.h"
#include "ppm/film.h"

namespace ppm {

struct Display : public beast::gl::async_window {
  Display(const Config& config, Film& film);
  void render();
  void set_captions(stringstream& header_ss, stringstream& footer_ss);

  bool is_on();

private:
  const Config& config;
  Film& film;
  string header;
  string footer;
  bool on;

  void print_string(void* font, const string& str) const;
  void keyboard(unsigned char key, int mousex, int mousey);
};

}

#endif // _PPM_DISPLAY_H_
