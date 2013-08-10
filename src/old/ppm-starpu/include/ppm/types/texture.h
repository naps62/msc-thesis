#ifndef _PPM_TYPES_TEXTURE_H_
#define _PPM_TYPES_TEXTURE_H_

#include "utils/common.h"

namespace ppm {

struct TexMap {
  uint rgb_offset, alpha_offset;
  uint width, height;
};

ostream& operator<< (ostream& os, const TexMap& t);

}

#endif // _PPM_TYPES_TEXTURE_H_
