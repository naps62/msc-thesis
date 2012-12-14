/*
 * Particle.h
 *
 *  Created on: May 23, 2012
 *      Author: amariano
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

class Particle {
private:

	float convergence;
	int childs;

	int X;
	int Y;
	int Z;

public:
	Particle(float);

	void process();
	void scale();
	virtual ~Particle();
};

#endif /* PARTICLE_H_ */
