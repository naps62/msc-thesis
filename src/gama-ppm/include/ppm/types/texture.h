/*
 * texture.h
 *
 *  Created on: Mar 19, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_TYPES_TEXTURE_H_
#define _PPM_TYPES_TEXTURE_H_

namespace ppm {

struct TexMap {
	uint rgb_offset, alpha_offset;
	uint width, height;
};

ostream& operator<< (ostream& os, const TexMap& t) {
	return os << "TexMap[" << t.rgb_offset << "; " << t.alpha_offset << ", " << t.width << "; " << t.height << "]";
}

}

#endif // _PPM_TYPES_TEXTURE_H_
