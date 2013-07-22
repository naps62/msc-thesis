/*
 * dumper.cpp
 *
 *  Created on: Jan 16, 2013
 *      Author: naps62
 */

#include <beast/profile/dumper.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::stringstream;
using std::pair;
using std::make_pair;
using std::ifstream;
using std::vector;

#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

namespace beast { namespace profile {

	void dumper :: dump() const {
		_if_prof(
			stringstream keys;
			stringstream values;
			for (auto it = data.begin(); it != data.end(); ++it) {
				keys   << (it->first)  << ';';
				values << (it->second) << ';';
			}

			std::cerr << keys.str() << '\n' << values.str() << std::endl;
		)
	}

} }
