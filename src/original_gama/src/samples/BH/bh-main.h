/*
 * bh_main.h
 *
 *  Created on: Aug 7, 2012
 *      Author: ricardo
 */

#ifndef BH_MAIN_H_
#define BH_MAIN_H_


/*
 * mb-nbody-main.h
 *
 *  Created on: Aug 3, 2012
 *      Author: ricardo
 */

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
double time_total=0,tcur=0,tavg = -1, time_fc=.0f,time_up=.0f,tstart,tend;
double tmax=FLT_MIN,tmin=FLT_MAX;
float *dq;
unsigned int* count;

//unsigned step=0;

void main_loop();

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

#ifdef DISPLAY
	if(restart==true){
		genInput();
		gstep=-1;
		restart=false;
	}
#endif

	BoundingBoxKernel();

	TreeBuildingKernel();

	SummarizationKernel();

	SortKernel();

	tcur=getTimeMS();

	register int i;
	register int maxdepth;
	register float tmp;

	maxdepth = gmaxdepth;
	tmp = gradius;
	// precompute values that depend only on tree level
	dq[0] = tmp * tmp * itolsq;
	for (i = 1; i < maxdepth; i++) {
		dq[i] = dq[i - 1] * 0.25f;
		dq[i - 1] += epssq;
	}
	dq[i - 1] += epssq;

	if (maxdepth <= MAXDEPTH) {
		BHForce* fc = new BHForce(	gmass, gposx, gposy, gposz,
				gvelx, gvely, gvelz,
				gaccx, gaccy, gaccz,
				gchild,gsort,
				0,NBODIES,gstep,dq,NNODES,maxdepth,gradius,count);
		rs->submit(fc);
		rs->synchronize();
		delete fc;
	}


	tcur=getTimeMS() - tcur;
    
    //ForceCalculationKernel();
	IntegrationKernel();

	if(gstep >= 1) {
		tmax = max(tmax,tcur);
		tmin = min(tmin,tcur);
		time_total += tcur;
		tavg = (tavg * .5f + tcur*.5f);
	} else {
		tavg = tcur;
	}

#ifdef DISPLAY
	moveToVBO2();
	unsigned int size = NBODIES * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, (const GLvoid*) particles, GL_DYNAMIC_DRAW);
	aux_Draw();
#endif

	if(gstep==NTIMESTEPS-1) {
		unsigned int c;
		cudaMemcpy(&c,count,sizeof(unsigned int),cudaMemcpyDeviceToHost);
		printf("%lu\n",c);
		printf("(V) Avg time per frame (ms): %7.3f  (Max: %7.3f, Min: %7.3f, Tot: %7.3f Avg: %7.3f )\n",tavg,tmax,tcur,time_total, time_total/(float)(gstep-1));
		delete rs;
		cudaFree(dq);
		exit(0);
	}
}



int main(int argc, char* argv[]) {

	rs =  new RuntimeScheduler();
#ifdef DISPLAY
	if (0 == initGL(&argc, argv)) {
		return 0;
	}
#endif

	cudaHostAlloc((void**)&dq, MAXDEPTH*sizeof(float), cudaHostAllocPortable);
	cudaMalloc((void**)&count, sizeof(unsigned int));
	cudaMemset(count, 0x0, sizeof(unsigned int));
	particles = new Point3D[NBODIES];
#ifdef DISPLAY
	createVBO(&vbo, (const GLvoid*) particles);
	glutDisplayFunc(main_loop);
#endif

	init_bhcpu();
#ifdef DISPLAY
	glutMainLoop();
#else
	while(true)	{
		main_loop();
	}
#endif

	cudaFree(dq);
	cudaFree(count);
	delete rs;
	exit (0);
}




#endif /* BH_MAIN_H_ */
