#ifndef _PPM_GEOMETRY_RAYBUFFER_H_
#define _PPM_GEOMETRY_RAYBUFFER_H_

#include <vector>
using std::vector;

#include "luxrays/core/dataset.h"
#include "luxrays/core/geometry/ray.h"
#include "utils/common.h"
#include "ppm/geometry/ray.h"
#include "ppm/geometry/rayhit.h"
#include "ppm/math.h"

namespace ppm {

struct RayBuffer {

  /*
   * constructors
   */

  // default constructor
  __HYBRID__ RayBuffer(const size_t _max_size)
  : rays(_max_size), hits(_max_size), _size(0), max_size(_max_size) { }


  unsigned int add(const Ray& ray) {
    assert(_size < max_size);
    rays[_size] = ray;
    return _size++;
  }

  void reset() {
    memset(&rays[0], 0, sizeof(Ray)*_size);
    memset(&hits[0], 0, sizeof(RayHit)*_size);
    _size = 0;
  }

  unsigned int size() const {
    return _size;
  }

  bool full() const {
    return size() == max_size;
  }

  void intersect(luxrays::DataSet* data_set) {
    for(unsigned i(0); i < _size; ++i) {
      luxrays::Ray&    ray = static_cast<luxrays::Ray>(rays[i]);
      luxrays::RayHit& hit = static_cast<luxrays::RayHit>(hits[i]);
      hit.set_miss();
      data_set->Intersect(&ray, &hit);
    }
  }

  private:
  vector<Ray> rays;
  vector<RayHit> hits;
  unsigned int _size;
  unsigned int max_size;
};

}

#endif // _PPM_GEOMETRY_RAYBUFFER_H_
