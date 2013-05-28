/*
 * beast.cpp
 *
 *  Created on: Jan 17, 2013
 *      Author: naps62
 */

#include <beast/init.hpp>

#include <unistd.h>
#include <fcntl.h>

namespace beast {

void init() {
	// if on profiling mode, redirect all normal output to /dev/null
	_if_prof(
		profile::__out_fd  = dup(fileno(stdout));
		profile::__null_fd = open("/dev/null", O_WRONLY);
		dup2(fileno(stdout),     profile::__out_fd);
		dup2(profile::__null_fd, fileno(stdout));
	)
}

}
