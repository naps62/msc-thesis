/*
 * mb-nbody-main.h
 *
 *  Created on: Aug 3, 2012
 *      Author: ricardo
 */

#ifndef MB_NBODY_MAIN_H_
#define MB_NBODY_MAIN_H_

RuntimeScheduler* rs;
smartPtr<float> g_mass;
smartPtr<float> g_posx;
smartPtr<float> g_posy;
smartPtr<float> g_posz;
smartPtr<float> g_velx;
smartPtr<float> g_vely;
smartPtr<float> g_velz;
smartPtr<float> g_accx;
smartPtr<float> g_accy;
smartPtr<float> g_accz;
double time_total,tavg = -1, time_fc=.0f,time_up=.0f,tstart,tend;
double tmax=FLT_MIN,tmin=FLT_MAX;
unsigned step=0;

void main_loop();
void main_loop2();

#include <samples/BH/BHcpu.h>
#include <samples/MB-NBODY/display.h>

Point3D* particles;

void moveToVBO(){
#pragma parallel omp for
	for (unsigned long i=0; i<NBODIES; i++){
		particles[i].x=g_posx.get(i);
		particles[i].y=g_posy.get(i);
		particles[i].z=g_posz.get(i);
	}
}


void moveToVBO2(){
#pragma parallel omp for
	for (unsigned long i=0; i<NBODIES; i++){
		particles[i].x=gposx[i];
		particles[i].y=gposy[i];
		particles[i].z=gposz[i];
	}
}


void main_loop() {


	time_total=getTimeMS();
	time_fc=time_up=.0f;


	if(restart==true){
		genInput();
		step=0;
		restart=false;
	}

	ForceCalculation* fc = new ForceCalculation(g_mass, g_posx, g_posy, g_posz, g_velx, g_vely, g_velz, g_accx, g_accy, g_accz,0,NBODIES,step);
	tstart = getTimeMS();
	rs->submit(fc);
	rs->synchronize();
	tend = getTimeMS();
	time_fc+=(tend-tstart);

	delete fc;



	UpdatePosition* up = new UpdatePosition(g_mass, g_posx, g_posy, g_posz, g_velx, g_vely, g_velz, g_accx, g_accy, g_accz,0,NBODIES);
	tstart = getTimeMS();
	rs->submit(up);
	rs->synchronize();
	tend = getTimeMS();
	time_up+=(tend-tstart);

	delete up;


	time_total=getTimeMS() - time_total;

	tmax = max(tmax,time_total);
	tmin = min(tmin,time_total);

	if (tavg == -1)  tavg = time_total; else tavg = (tavg * 0.8f+ time_total*0.2f);

	printf("%u step -- Avg time: %.3fms (Force: %.3fms Update: %.3fms)\n",step,tavg, time_fc,time_up);



	moveToVBO();
	unsigned int size = NBODIES * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, (const GLvoid*) particles, GL_DYNAMIC_DRAW);
	aux_Draw();
	step++;
}



int main(int argc, char* argv[]) {

	rs =  new RuntimeScheduler();
	step=0;

	if (0 == initGL(&argc, argv)) {
		return 0;
	}


    particles = new Point3D[NBODIES];
    createVBO(&vbo, (const GLvoid*) particles);

    glutDisplayFunc(main_loop);
    init_bhcpu();

    g_mass = smartPtr<float>(sizeof(float)*NBODIES);
    g_posx = smartPtr<float>(sizeof(float)*NBODIES);
    g_posy = smartPtr<float>(sizeof(float)*NBODIES);
    g_posz = smartPtr<float>(sizeof(float)*NBODIES);
    g_velx = smartPtr<float>(sizeof(float)*NBODIES);
    g_vely = smartPtr<float>(sizeof(float)*NBODIES);
    g_velz = smartPtr<float>(sizeof(float)*NBODIES);
    g_accx = smartPtr<float>(sizeof(float)*NBODIES);
    g_accy = smartPtr<float>(sizeof(float)*NBODIES);
    g_accz = smartPtr<float>(sizeof(float)*NBODIES);

    genInput();

	glutMainLoop();

	delete rs;
}


#endif /* MB_NBODY_MAIN_H_ */
