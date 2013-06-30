/*
 * debug.hpp
 *
 *  Created on: Jan 15, 2013
 *      Author: naps62
 */

#ifndef _BEAST_DEBUG_HPP_
#define _BEAST_DEBUG_HPP_

/*
 * Defines a couple of helpers that are only enabled on Debug compilations (where _Debug is defined)
 *
 * 	_debug
 * 		evaluates to true if debug mode is enabled
 *
 * 	_if_dbg(x)
 * 		(x) is only evaluated in debug mode
 *
 * 	_if_not_dbg(x)
 * 		(x) is only evaluated outside of debug mode
 *
 * 	example:
 * 		if (_debug) {
 * 			std::cout << "this is debug mode"
 * 		}
 *
 * 		is mostly equivalent to:
 *
 * 		_if_dbg(
 * 			std::cout << "this is debug mode"
 * 		)
 *
 * 		with the exception that the latter doesn't introduce any overhead for non-debug compilations,
 * 		and also doesn't evaluate the if condition in run-time for debug compilations
 */
#ifdef _Debug
	#define _debug			true
	#define _if_dbg(x)		x
	#define _if_not_dbg(x)
#else
	#define _debug			false
	#define _if_dbg(x)
	#define _if_not_dbg(x)	x
#endif // _Debug


#endif // _BEAST_DEBUG_HPP_
