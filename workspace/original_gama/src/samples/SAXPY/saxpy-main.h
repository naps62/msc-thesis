/*
 * saxpy-main.h
 *
 *  Created on: Aug 3, 2012
 *      Author: ricardo
 */

#ifndef SAXPY_MAIN_H_
#define SAXPY_MAIN_H_

#include <cublas.h>

int main(int argc, char* argv[]) {

	RuntimeScheduler* rs =  new RuntimeScheduler();

	smartPtr<float> R = smartPtr<float>(sizeof(float)*N);
	smartPtr<float> X = smartPtr<float>(sizeof(float)*N);
	smartPtr<float> Y = smartPtr<float>(sizeof(float)*N);
	float alpha = 1.5f;

	srand(7);
	for(int i=0; i < N; i++) {
		R[i] = 0.f;
		X[i] = 1.5f; //float(rand())/RAND_MAX+1;
		Y[i] = float(rand())/RAND_MAX+1;
	}



//
	for(int i=0; i < N; i++) {
			R[i] = 0.f;
	}

	saxpy* s = new saxpy(R,X,Y,alpha,N,0,N);

	rs->synchronize();
	double start = getTimeMS();
	rs->submit(s);
	rs->synchronize();
	double end = getTimeMS();

	unsigned int count = 0;

	for(int i=0; i <  N; i++) {
		if(fabs(R[i] -( X[i]*alpha+Y[i])) > FLT_EPSILON*4) {
			count++;
		}
	}

	double start_m = getTimeMS();
		cublasSaxpy(N,alpha,X.getPtr(),1,Y.getPtr(),1);
		cudaDeviceSynchronize();
		double end_m = getTimeMS();
		printf("Time blas Kernel: %.4f\n",(end_m-start_m));

	printf("(V) Time GAMA: %.4f Error: %d (%d) -- %.3f\% \n",(end-start),count,N,(float(count)/float(N))*100);

    delete rs;
}

#endif /* SAXPY_MAIN_H_ */
