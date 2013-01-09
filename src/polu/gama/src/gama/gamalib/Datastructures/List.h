/*
 * List
 *
 *  Created on: Aug 1, 2011
 *      Author: jbarbosa
 */

#ifndef LIST_
#define LIST_

template <typename T> class List {

public:

	T* array;
	unsigned int _size;


public:

	__HYBRID__ List(unsigned int size=1) : _size(size) {
		array = new T[_size];
	}

	__HYBRID__ ~List() {
		delete array;
	}


	__HYBRID__ unsigned long getSize() {
		return _size;
	}

	__HYBRID__ T &operator[] (const int index) {
		return array[index];
	}

};



#endif /* LIST_ */
