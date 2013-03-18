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
	__HYBRID__ __forceinline vector()
	: smartPtr<T>(size_t(0)) {
		n(sizeof(uint));
		n.set(0, 0);
	}

	// allocs space for `size` elements of type T
	__HYBRID__ __forceinline vector(const uint size)
	: smartPtr<T>(size * sizeof(T)) {
		n(sizeof(uint));
		n[0] = size;
	}

	// copy constructor
	__HYBRID__ __forceinline vector(vector<T>& copy)
	: smartPtr<T>(copy) {
		n(sizeof(uint));
		n[0] = copy.size();
	}

	/*
	 * methods
	 */

	// gets the current size of the array, in units of sizeof(T)
	__HYBRID__ __forceinline uint size() {
		return n[0];
	}

	// override operator() to support implicit sizeof(T) in space allocation
	// original operator asks only for total number of bytes
	__HYBRID__ __forceinline void operator() (const uint size) {
		this->alloc(size);
	}

	// reallocates the array, but backs up all possible data
	// if the new array size is lower than previous one, data at the end is discarded
	// TODO this is not HYBRID. Test this
	void resize(const uint size) {
		// backup old data
		const uint old_n   = this->n[0];
		const T*   old_ptr = this->ptr;

		// realloc
		this->alloc(size);

		// copy as much data as possible
		const uint copy_size = min(old_n, size);
		memcpy(this->ptr, old_ptr, copy_size * sizeof(T));
	}

	// increments size of the array, and places new elem at the end
	void push_back(const T& t) {
		const uint new_n = n[0] + 1;
		resize(new_n);
		this->set(n[0], t);
		n[0] = new_n;
	}

private:
	smartPtr<uint> n; // number of T elements currently allocated

	__HYBRID__ __forceinline void alloc(const uint size) {
		smartPtr<T>::alloc(size * sizeof(T));
		n[0] = size;
	}
};

}

#endif // _GAMA_EXT_VECTOR_H_
