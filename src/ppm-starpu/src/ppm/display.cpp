#include "ppm/display.h"
#include <GL/freeglut.h>

namespace ppm {

//
// constructors
//

Display :: Display(const Config& _config, Film& _film)
: beast::gl::async_window(_config.title, _config.width, _config.height), config(_config), film(_film), on(true) {
}

//
// public methods
//

void Display :: render() {
  cout << "drawing" << endl;
  glRasterPos2i(0, 0);
  // draw frame buffer
  film.UpdateScreenBuffer();
  glDrawPixels(film.width, film.height, GL_RGB, GL_FLOAT, film.GetScreenBuffer());

  // draw caption background
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColor4f(0.f, 0.f, 0.f, 0.5f);
  glRecti(0, 0, config.width - 1, 18);
  glRecti(0, config.height - 15, config.width - 1, config.height - 1);
  glDisable(GL_BLEND);

//  // header
  glColor3f(1.f, 1.f, 1.f);
  glRasterPos2i(4, config.height - 11);
  print_string(GLUT_BITMAP_8_BY_13, header);
//
//  // footer
  glRasterPos2i(4, 5);
  print_string(GLUT_BITMAP_8_BY_13, footer);
}

bool Display :: is_on() {
  return on;
}

void Display :: print_string(void *font, const string& str) const {
  for (uint i(0); i < str.length(); i++)
    glutBitmapCharacter(font, str[i]);
}

void Display :: set_captions(stringstream& header_ss, stringstream& footer_ss) {
  header = header_ss.str();
  footer = footer_ss.str();
}

void Display :: keyboard(unsigned char key, int mousex, int mousey) {
  switch(key) {
    case 27: // ESC
      on = false;
      break;
  }
}

}
