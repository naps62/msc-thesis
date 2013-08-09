#ifndef _PPM_TYPES_PATHS_H_
#define _PPM_TYPES_PATHS_H_

#include "utils/common.h"

namespace ppm {

struct EyePath {
  float scr_x, scr_y;
  Ray ray;
  uint depth;
  Spectrum flux;
  bool done;
  bool splat;
  uint sample_index;

  __HD__
  EyePath()
  : scr_x(0), scr_y(0), ray(), depth(0), done(false), splat(false) { }
};

struct PhotonPath {
  Ray ray;
  Spectrum flux;
  uint depth;
  bool done;

  __HD__
  PhotonPath()
  : ray(), flux(1.f, 1.f, 1.f), depth(0), done(false) { }
};

}

#endif // _PPM_TYPES_PATHS_H_
