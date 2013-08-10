/*
 * profile.hpp
 *
 *  Created on: Jan 12, 2013
 *      Author: naps62
 */

#ifndef _BEAST_PROF_HPP_
#define _BEAST_PROF_HPP_

#include <unistd.h>
#include <iostream>
#include <fstream>

/*
 * Defines a couple of helpers that are only enabled on Profile compilations (where _Profile is defined)
 *
 * 	_profile
 * 		evaluates to true if profile mode is enabled
 *
 * 	_if_prof(x)
 * 		(x) is only evaluated in profile mode
 *
 * 	_if_not_prof(x)
 * 		(x) is only evaluated outside of profile mode
 *
 * 	example:
 * 		if (_profile) {
 * 			std::cout << "this is profile mode"
 * 		}
 *
 * 		is mostly equivalent to:
 *
 * 		_if_prof(
 * 			std::cout << "this is profile mode"
 * 		)
 *
 * 		with the exception that the latter doesn't introduce any overhead for non-profile compilations,
 * 		and also doesn't evaluate the if condition in run-time for profile compilations
 */
#ifdef _Profile
	#define _profile        true
	#define _if_prof(x)     x
	#define _if_not_prof(x)
#else
	#define _profile        false
	#define _if_prof(x)
	#define _if_not_prof(x) x
#endif // _Profile

namespace beast { namespace profile {
	_if_prof(
		extern int __out_fd;
		extern int __null_fd;

		struct __out_class : public std::ofstream {
			template<class T>
			__out_class& operator << (const T& obj) {
				dup2(profile::__out_fd,  fileno(stdout)); // re-enable stdout
				std::cout << obj;
				std::cout.flush();
				dup2(profile::__null_fd, fileno(stdout));  // disable stdout again
				return *this;
			}
		};

		extern __out_class out;
	)
} }

#include <beast/profile/defaults.hpp>
#include <beast/profile/dumper.hpp>

#endif // _BEAST_PROF_HPP_
