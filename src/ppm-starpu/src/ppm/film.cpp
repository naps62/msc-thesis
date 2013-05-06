#include "ppm/film.h"

namespace ppm {

Film :: Film(const Config& _config)
: config(_config) {
  width = config.width;
  height = config.height;
  image_buffer = std::vector<Spectrum>(width*height);
  frame_buffer = std::vector<Spectrum>(width*height);
  has_changed = false;
}

void Film :: clear(const Spectrum color) {
  boost::recursive_mutex::scoped_lock lock(buffer_mutex);

  for(uint i = 0; i < image_buffer.size(); ++i) {
    image_buffer[i] = color;
  }
  has_changed = true;
}

Spectrum* Film :: get_frame_buffer_ptr() {
  update_frame_buffer();
  return &frame_buffer[0];
}

void Film :: update_frame_buffer() {
  boost::unique_lock<boost::recursive_mutex> lock(buffer_mutex, boost::defer_lock_t());
  if (config.vsync)
    lock.lock();

  if (!has_changed)
    return;

  // TODO when using ppma or sppma, weight is not always 1
  const float weight = 1;
  const float inv_weight = 1 / weight;
  for(uint i = 0; i < image_buffer.size(); ++i) {
    frame_buffer[i] = image_buffer[i] * inv_weight;
  }
  has_changed = false;
}

}
