/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <beast/debug.hpp>
#include <beast/profile.hpp>

#include <iostream>
int main() {

	_debug(std::cout << "debug \n");
	_profile(std::cout << "profile\n");

}
