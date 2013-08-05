#ifndef _PPM_TYPES_PATHS_H_
#define _PPM_TYPES_PATHS_H_

#include "utils/common.h"

namespace ppm {

struct Path {
  Spectrum flux;
  Ray ray;
  uint depth;
  bool done;

  __HD__
  Path()
  : flux(1.f, 1.f, 1.f), ray(), depth(0), done(false) { }
};

struct EyePath : Path {
  float scr_x, scr_y;
  bool splat;
  uint sample_index;

  __HD__
  EyePath()
  : Path(), scr_x(0), scr_y(0), splat(false) { }
};

struct PhotonPath : Path {

};

}

#endif // _PPM_TYPES_PATHS_H_
