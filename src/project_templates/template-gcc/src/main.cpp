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

	_if_dbg(std::cout << "debug \n");
	_if_prof(std::cout << "profile\n");

}
