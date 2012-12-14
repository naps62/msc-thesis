/*
 * Particle.cpp
 *
 *  Created on: May 23, 2012
 *      Author: amariano
 */

#include "Particle.h"
#include <stdlib.h>

Particle::Particle(float kids) {

	childs = rand()%10 + 1;

	X = rand()%10;
	Y = rand()%10;
	Z = rand()%10;

}

void Particle::process() {

	long random = rand()%10000000 + 1;

	for(int i = 0; i < random ; i++) scale();

}

void Particle::scale() {

	X *= 1.1;
	Y *= 1.2;
	Z *= 1.3;

}

Particle::~Particle() {

}
