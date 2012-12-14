/*
 * gemm-main.h
 *
 *  Created on: Aug 3, 2012
 *      Author: ricardo
 */

#ifndef GEMM_MAIN_H_
#define GEMM_MAIN_H_

int main(int argc, char* argv[]) {

	RuntimeScheduler* rs =  new RuntimeScheduler();
	//MemorySystem *mem = new MemorySystem();
	//LowLevelMemAllocator::_memSys=mem;

	smartPtr<float> A = smartPtr<float>(sizeof(float)*NN*NN, SHARED);
	smartPtr<float> B = smartPtr<float>(sizeof(float)*NN*NN, SHARED);
	smartPtr<float> C = smartPtr<float>(sizeof(float)*NN*NN, SHARED);

	for(int i=0; i < NN*NN; i++) {
		A[i] = 4;
	}

	for(int i=0; i < NN; i++) {

		for(int j=0; j < NN; j++) {

			if(i==j){

				B[i*NN+j] = 1;

			}
			else{

				B[i*NN+j] = 0;

			}

		}

	}

//	printf("Matriz Identidade: \n");
//
//	for(int i=0; i < NN; i++) {
//
//		for(int j=0; j < NN; j++) {
//
//			printf("%f ",B[i*NN+j]);
//
//		}
//
//		printf("\n");
//
//	}

	for(int i=0; i < NN*NN; i++) {
		C[i] = 0;
	}

	gemm* g = new gemm(A,B,C,NN,0,0);

	rs->synchronize();
	double start = getTimeMS();
	rs->submit(g);
	rs->synchronize();
	double end = getTimeMS();

	unsigned int count = 0;

	for(int i=0; i <  NN*NN; i++) {
		if(C[i] != 4.f) {count++; }//printf("Error pos: %d, %f\n",i,C[i]);}
	}

	printf("(V) Time GAMA: %.4f Error: %d\n",(end-start),count);

//	double start_cu = getTimeMS();
//	cublasSgemm('N','N',N,N,N,1,A.getPtr(),N,B.getPtr(),N,1,C.getPtr(),N);
//	cuCtxSynchronize();
//
//	printf("Matriz C: \n");
//
//		for(int i=40; i < 50; i++) {
//
//			for(int j=50; j < 60; j++) {
//
//				printf("%f ",C[i*N+j]);
//
//			}
//
//			printf("\n");
//
//		}
//
//	double end_cu = getTimeMS();
//	printf("Time cuBlas: %.4f\n",(end_cu-start_cu));
//	fflush(stdout);X86MemFence();

	delete rs;
}

#endif /* GEMM_MAIN_H_ */
