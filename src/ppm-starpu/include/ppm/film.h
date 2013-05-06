#ifndef _PPM_FILM_H_
#define _PPM_FILM_H_

#include "utils/common.h"
#include "utils/config.h"

namespace ppm {

class Film {

public:
  Film(const Config& _config);
  void clear(const Spectrum color = Spectrum());
  Spectrum* get_frame_buffer_ptr();
  void update_frame_buffer();

public:
  const Config& config;
  uint width, height;
  boost::recursive_mutex buffer_mutex;
  std::vector<Spectrum> image_buffer;
  std::vector<Spectrum> frame_buffer;
  bool has_changed;
};

}

#endif // _PPM_FILM_H_
