/*
 * profiler.hpp
 *
 *  Created on: Jan 12, 2013
 *      Author: naps62
 */

#ifndef _BEAST_PROF_DUMPER_HPP_
#define _BEAST_PROF_DUMPER_HPP_

#include <beast/profile.hpp>

#include <vector>
#include <map>
#include <string>
#include <sstream>

using std::vector;
using std::map;
using std::string;
using std::stringstream;

namespace beast { namespace profile {

	/*
	 * Defines a map<string, string> of profiling values to dump
	 */
	class dumper {

		// key-value pair containing counter name and value
		map<const string, const string> data;

	public:

		// adds a new abstract object to the map
		// assumes obj has a name() and a value() method
		template<class T>
		void add(const T& obj);

		// dumps the entire map in CSV format
		void dump() const;
	};

	/*
	 * template definitions
	 */
	template<class T>
	void dumper :: add(const T& obj) {
		_if_prof(
			stringstream name;
			stringstream value;
			name  << obj.name();
			value << obj.value();
			data.insert(make_pair(name.str(), value.str()));
		)
	}

} }

#endif // _BEAST_PROF_DUMPER_HPP_
