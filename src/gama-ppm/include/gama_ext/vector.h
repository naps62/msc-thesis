/*
 * vector.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _GAMA_EXT_VECTOR_H_
#define _GAMA_EXT_VECTOR_H_

#include <gama/gamalib/memlib/smartpointer.h>

namespace gama {

/*
 * an extension over GAMA's smartPtr, providing a more extensive vector interface
 */
template<class T>
struct vector : public smartPtr<T> {

	/*
	 * constructors
	 */

	// empty construtor (no allocation)
//	__HYBRID__ __forceinline vector();

	// allocs space for `size` elements of type T
	__HYBRID__ __forceinline vector(const uint size)
	: smartPtr<T>(size * sizeof(T)) {
		n = size;
	}

	// copy constructor
	__HYBRID__ __forceinline vector(const vector<T>& copy)
	: smartPtr<T>(copy) {
		n = copy.size();
	}

	/*
	 * methods
	 */

	// gets the current size of the array, in units of sizeof(T)
	__HYBRID__ __forceinline uint size() const {
		return n;
	}

	// override operator() to support implicit sizeof(T) in space allocation
	// original operator asks only for total number of bytes
	__HYBRID__ __forceinline void operator() (const uint size) {
		this->realloc(size);
	}

	// reallocates entire vector, without persisting data
	__HYBRID__ __forceinline void realloc(const uint size) {
		this->alloc(size);
		n = size;
	}

	// reallocates the array, but backs up all possible data
	// if the new array size is lower than previous one, data at the end is discarded
	// TODO this is not HYBRID. Test this
	void resize(const uint size) {
		// backup old data
		const uint old_n   = this->n;
		const T*   old_ptr = this->ptr;

		// realloc
		this->alloc(size);

		// copy as much data as possible
		const uint copy_size = min(old_n, size);
		memcpy(this->ptr, old_ptr, copy_size * sizeof(T));
	}

private:
	uint n; // number of T elements currently allocated
};

}

#endif // _GAMA_EXT_VECTOR_H_
