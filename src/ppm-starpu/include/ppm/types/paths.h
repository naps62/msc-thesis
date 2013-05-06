#ifndef _PPM_TYPES_PATHS_H_
#define _PPM_TYPES_PATHS_H_

#include "utils/common.h"

namespace ppm {

struct Path {
  Spectrum flux;
  uint depth;
  bool done;

  Path()
  : flux(1.f, 1.f, 1.f), depth(0), done(false) { }
};

struct EyePath : Path {
  float scr_x, scr_y;
  Ray ray;
  bool splat;
  uint sample_index;

  EyePath()
  : Path(), scr_x(0), scr_y(0), ray(), splat(false) { }
};

struct PhotonPath : Path {

};

}

#endif // _PPM_TYPES_PATHS_H_
