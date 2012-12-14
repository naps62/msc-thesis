/*
 * fft-main.h
 *
 *  Created on: Aug 3, 2012
 *      Author: ricardo
 */

#ifndef FFT_MAIN_H_
#define FFT_MAIN_H_

#include <fftw3.h>
#include <cufft.h>
#define BATCH 1

// Main syntha:

//int main(int argc, char* argv[]) {
//
//	RuntimeScheduler* rs =  new RuntimeScheduler();
//
//	syntha* s = new syntha(0);
//
//	rs->synchronize();
//	double start = getTimeMS();
//	rs->submit(s);
//	rs->synchronize();
//	double end = getTimeMS();
//
//	printf("Time GAMA: %.4f\n",(end-start));
//
//	delete rs;
//}

void bitTableF(smartPtr<int> reversalTable){

	unsigned int nforward, nreversed;
	unsigned int count;

	unsigned int logN = log2((float)ARRAY_SIZE*2);

	for( int i = 0; i<ARRAY_SIZE*2; i+=2 ){

		nreversed = i;
		count = logN-1;

		for(nforward=i>>1; nforward; nforward>>=1){

		   nreversed <<= 1;
		   nreversed |= nforward & 1; 	// give LSB of nforward to nreversed
		   count--;

		}

		nreversed <<=count; 			// compensation for missed iterations
		nreversed &= ARRAY_SIZE-1;   			// clear all bits more significant than N-1

		reversalTable[i] = (nreversed*2);
		reversalTable[i+1] = (nreversed*2)+1;

	}

}

void my_NthRoots(smartPtr<double> nthrootsDataR, smartPtr<double> nthrootsDataI){

	int stage, j, _end= log2((float)ARRAY_SIZE), __end, i, summedStage;

	double argV = -2*MPI;
	double argV2, argV3;

	for(stage=2,j=0;j<_end;stage=stage<<1,j++){
		__end=stage/2;
		argV2 = argV/stage;
		for(i=0,summedStage=stage;i<__end;summedStage++,i++){
			argV3 = argV2*i;
			nthrootsDataR[summedStage] = cos(argV3);
			nthrootsDataI[summedStage] = sin(argV3);
		}
	}

}

// Main FFT:

int main(int argc, char* argv[]) {

	RuntimeScheduler* rs =  new RuntimeScheduler();

		smartPtr<double> dataInput = smartPtr<double>(sizeof(double)*ARRAY_SIZE*2, SHARED);

		smartPtr<double> dataORD = smartPtr<double>(sizeof(double)*ARRAY_SIZE*2, SHARED);
		smartPtr<double> dataOutput = smartPtr<double>(sizeof(double)*ARRAY_SIZE*2, SHARED);

		smartPtr<int> reversalTable = smartPtr<int>(sizeof(int)*ARRAY_SIZE*2, SHARED);
		smartPtr<int> reversalTable2 = smartPtr<int>(sizeof(int)*ARRAY_SIZE*2, SHARED);

		smartPtr<double> nthrootsdataR = smartPtr<double>(sizeof(double)*(ARRAY_SIZE+(ARRAY_SIZE/2)), SHARED);
		smartPtr<double> nthrootsdataI = smartPtr<double>(sizeof(double)*(ARRAY_SIZE+(ARRAY_SIZE/2)), SHARED);

		if( !dataInput.getPtr() && !dataORD.getPtr() ) {
			printf("ERROR ON THE INITialization\n");
			exit(0);
		}

		/* FFT input initialization */

		for (int i = 0; i < ARRAY_SIZE*2; i+=2) {

			dataInput[i] = 0.5;
			dataInput[i+1] = 0;//newComplex;

		}

		double mcopy0 = getTimeMS();
		//memcpy(dataORD.getPtr(),dataInput.getPtr(),2*8*ARRAY_SIZE);
		double mcopy1 = getTimeMS();

		printf( "Time mem copy = %.4f\n" ,mcopy1-mcopy0);

		if(ARRAY_SIZE==16){
			dataInput[0] = 9.34;
			dataInput[1] =  0.0;
			dataInput[2] = 60.2;
			dataInput[3] =  000.0;
			dataInput[4] = 55.3;
			dataInput[5] =  000.0;
			dataInput[6] = 12.21;
			dataInput[7] =  000.0;
			dataInput[8] = 64.0;
			dataInput[9] =  000.0;
			dataInput[10] = 1.0;
			dataInput[11] =  000.0;
			dataInput[12] = 99.0;
			dataInput[13] =  000.0;
			dataInput[14] = 100.0;
			dataInput[15] =  000.0;
			dataInput[16] = 8.00;
			dataInput[17] =  000.0;
			dataInput[18] = 9.0;
			dataInput[19] =  000.0;
			dataInput[20] = 10.0;
			dataInput[21] =  0.0;
			dataInput[22] = 11.0;
			dataInput[23] =  000.0;
			dataInput[24] = 12.0;
			dataInput[25] =  000.0;
			dataInput[26] = 13.0;
			dataInput[27] = 000.0;
			dataInput[28] = 14.0;
			dataInput[29] =  000.0;
			dataInput[30] = 15.88;
			dataInput[31] =  000.0;
		}

		printf("Computing the FFT with %d elements!\n",ARRAY_SIZE);

		///////////////////////////////////////// Preparation phase (NOT included in the elapsed time)

		double start_prep2 = getTimeMS();
		//GAMA version:
		/*
		bitTable* bt = new bitTable(reversalTable,0,0,ARRAY_SIZE);
		rs->submit(bt);
		rs->synchronize();
		*/
		//Sequential version:
		bitTableF(reversalTable);
		double end_prep2 = getTimeMS();

		int counter, skip = 0;

		double start_prep = getTimeMS();
		//GAMA version:
		/*
		nthrootsKernel* nth = new nthrootsKernel(nthrootsdataR,0,0,nthrootsdataI,ARRAY_SIZE,0);//,0,0);

		rs->submit(nth);
		rs->synchronize();
		*/
		//Sequential version:
		my_NthRoots(nthrootsdataR,nthrootsdataI);
		double end_prep = getTimeMS();

		if(ARRAY_SIZE==16){
			for(int i=0;i<(ARRAY_SIZE+ARRAY_SIZE/2);i++)
				printf("%d = (%f,%f)\n",i,nthrootsdataR[i],nthrootsdataI[i]);

			getchar();
		}

		printf("Preparation phase completed in %.4f + %.4f miliseconds! \n", (end_prep-start_prep), (end_prep2-start_prep2) );

		/* Offset construction (essential step!) */

		double G00 = getTimeMS();

		int i,j=0,k,workersBR;

		if(ARRAY_SIZE==16) workersBR = 8;
		else workersBR = 128;

		int chunksize = ARRAY_SIZE/workersBR;

		int *offset = (int*)malloc(workersBR*sizeof(int));

		for(i=0;i<workersBR;i++){
			if(i==0) j=0;
			else{
				k = workersBR/2;
				while(k<=j){
					j=j-k;
					k=k/2;
				}
				j=j+k;
			}
		offset[i]=j;
		}
		double G01 = getTimeMS();

		///////////////////////////////////////// BitReversal

		bitKernel* w = new bitKernel(reversalTable,dataInput,0,0,dataORD,0,0,offset,chunksize,0,0,ARRAY_SIZE,workersBR);

		//printf("Init Primeiro kernel...\n");

		double start1 = getTimeMS();
		rs->submit(w);
		//Necessary to measure the kernel elapsed time:
		rs->synchronize();
		double end1 = getTimeMS();

		printf("Bit Reversal done in %.4f miliseconds! \n",(end1-start1) );

		if(ARRAY_SIZE==16){
			for(int  i = 0, o = 0 ; i < ARRAY_SIZE*2 ; i+=2, o++ ) {
				printf("Elem(%d) = (%f,%f);\n",o,dataORD[i],dataORD[i+1]);
			}
			getchar();
		}



		//*************************************************************************  Butterflies  *************************************************************************

		double start3 = getTimeMS();
		int totalStages = (int)log2((float)ARRAY_SIZE);

	    ///////////////////////////////////////// Kernel Primeiro Stage

		int stage=1, wingsize;

		for(wingsize=1;wingsize<chunksize;wingsize=wingsize<<1,stage++){

			//start3 = getTimeMS();
			firstSetKernel* set1 = new firstSetKernel(dataORD,0,0,nthrootsdataR,nthrootsdataI,wingsize,chunksize,0,0);

			rs->submit(set1);
			rs->synchronize();
			//double _end3 = getTimeMS();

			//printf(">> First Butterfly Set ++ in %.4f miliseconds! \n",(_end3-start3) );

			//break;
			printf("FirstSetKernel++\n");

		}
		double end3 = getTimeMS();

		printf("First Butterfly Set done in %.4f miliseconds! \n",(end3-start3) );

		if(ARRAY_SIZE==16){
			for(int  i = 0, o = 0 ; i < ARRAY_SIZE*2 ; i+=2, o++ ) {
				printf("Elem(%d) = (%f,%f);\n",o,dataORD[i],dataORD[i+1]);
			}
			getchar();
		}

		//printf("Second Butterfly Set on the way! \n" );

		///////////////////////////////////////// Kernel Segundo Stage

		double start4 = getTimeMS();
		int butterflies = ARRAY_SIZE;
		int butterfliesPerThread = butterflies/workersBR;

	    stage--;

	    if(stage+1<totalStages){
			int threadsPerBlock;
			stage++;
			for(threadsPerBlock=2 /*,chunksize*=2*/ ;stage!=totalStages;stage++,wingsize*=2,threadsPerBlock*=2 /*,chunksize*=2*/ ){

				middleSetKernel* set2 = new middleSetKernel(dataORD,0,nthrootsdataR,nthrootsdataI,wingsize,chunksize,butterfliesPerThread,threadsPerBlock,0,0);

				rs->submit(set2);
				rs->synchronize();

				printf("MiddleSetKernel++ \n");

				if(ARRAY_SIZE==16){
					for(int  i = 0, o = 0 ; i < ARRAY_SIZE*2 ; i+=2, o++ ) {
						printf("Elem(%d) = (%f,%f);\n",o,dataORD[i],dataORD[i+1]);
					}
					getchar();
				}


			}
			stage--;
	    }
	    double end4 = getTimeMS();

	    printf("Second Butterfly Set done in %.4f miliseconds! \n",(end4-start4) );



	    ///////////////////////////////////////// Kernel Terceiro Stage (Testado)

		double start5 = getTimeMS();
		int index = butterflies/workersBR;

		stage=totalStages-1;

		if(stage==totalStages-1){

			stage++;

			lastSetKernel* set3 = new lastSetKernel(dataORD,dataOutput,0,nthrootsdataR,nthrootsdataI,wingsize,chunksize,chunksize,index,butterfliesPerThread,0);

			rs->submit(set3);

		}

		rs->synchronize();

		double end5 = getTimeMS();

		printf("Third Butterfly Set done in %.4f miliseconds! \n",(end5-start5) );

		if(ARRAY_SIZE==16){
			printf("wingsize = %d\n",wingsize);
			for(int  i = 0, o = 0 ; i < ARRAY_SIZE*2 ; i+=2, o++ ) {
				printf("Elem(%d) = (%f,%f);\n",o,dataOutput[i],dataOutput[i+1]);
			}
			getchar();
		}

		double Gend = getTimeMS();


		long long int N = ARRAY_SIZE;

		double startfftw1 = getTimeMS();
		fftw_complex *in, *out;

		fftw_plan p;

		in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
		out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

		double endfftw1 = getTimeMS();

		for(int i=0;i<N;i++){
			in[i][0] = 0.5;
			in[i][1] = 0;
		}

		if(ARRAY_SIZE==16){
			in[0][0] = 9.34;
			in[0][1] =  000.0;
			in[1][0]  = 60.2;
			in[1][1]  =  000.0;
			in[2][0]  = 55.3;
			in[2][1]  =  000.0;
			in[3][0]  = 12.21;
			in[3][1]  =  000.0;
			in[4][0]  = 64.0;
			in[4][1]  =  000.0;
			in[5][0]  = 1.0;
			in[5][1]  =  000.0;
			in[6][0]  = 99.0;
			in[6][1]  =  000.0;
			in[7][0]  = 100.0;
			in[7][1]  =  000.0;
			in[8][0]  = 8.00;
			in[8][1]  =  000.0;
			in[9][0]  = 9.0;
			in[9][1]  =  000.0;
			in[10][0]  = 10.0;
			in[10][1]  =  000.0;
			in[11][0]  = 11.0;
			in[11][1]  =  000.0;
			in[12][0]  = 12.0;
			in[12][1]  =  000.0;
			in[13][0]  = 13.0;
			in[13][1]  = 000.0;
			in[14][0]  = 14.0;
			in[14][1]  =  000.0;
			in[15][0]  = 15.88;
			in[15][1]  =  000.0;
		}

		double startfftw2 = getTimeMS();
		p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(p); /* repeat as needed */
		double endfftw2 = getTimeMS();

		long errors = 0;

		double a,b,c,d;

		for(int  i = 0, j = 0 ; i < ARRAY_SIZE ; i++ ) {

			a = (int)floor(dataOutput[j]+0.5);
			b = (int)floor(out[i][0]+0.5);

			c = (int)floor(dataOutput[j+1]+0.5);
			d = (int)floor(out[i][1]+0.5);

			if(a != b){
				//printf("ERRO --- (%f,%f)",a,b);
				errors++;
			}
			if(c != d) errors++;

			j+=2;

		}

		if(ARRAY_SIZE==16){

			printf("FFTW:\n\n");

			for(int  i = 0 ; i < 16 ; i++ ) {
				printf("Elem(%d) = (%f,%f);\n",i,out[i][0],out[i][1]);
			}
		}

		fftw_destroy_plan(p);
		fftw_free(in);
		fftw_free(out);

	    double startcufft = getTimeMS();
		cufftHandle plan;

		cufftComplex *d_data;

		cufftComplex *data = (cufftComplex*)malloc(sizeof(cufftComplex)*ARRAY_SIZE*BATCH);

		for(int i=0;i<ARRAY_SIZE;i++){
			data[i] = make_cuComplex(0.5,0);
		}

		if(ARRAY_SIZE==16){
			data[0] = make_cuComplex(9.34,0);
	        data[1] = make_cuComplex(60.2,0);
	        data[2] = make_cuComplex(55.3,0);
	        data[3] = make_cuComplex(12.21,0);
	        data[4] = make_cuComplex(64.0,0);
	        data[5] = make_cuComplex(1.0,0);
	        data[6] = make_cuComplex(99.0,0);
	        data[7] = make_cuComplex(100.0,0);
	        data[8] = make_cuComplex(8.00,0);
	        data[9] = make_cuComplex(9.0,0);
	        data[10] = make_cuComplex(10.0,0);
	        data[11] = make_cuComplex(11.0,0);
	        data[12] = make_cuComplex(12.0,0);
	        data[13] = make_cuComplex(13.0,0);
	        data[14] = make_cuComplex(14.0,0);
	        data[15] = make_cuComplex(15.88,0);
		}

		cudaMalloc((void**)&d_data,sizeof(cufftComplex)*ARRAY_SIZE*BATCH);

		cudaMemcpy(d_data, data, BATCH*ARRAY_SIZE*sizeof(cufftComplex), cudaMemcpyHostToDevice);

		if( cufftPlan1d (&plan , ARRAY_SIZE , CUFFT_C2C , BATCH ) != CUFFT_SUCCESS ) {
			printf("CUFFT error\n");
		}

		if ( cufftExecC2C ( plan , d_data , d_data , CUFFT_FORWARD ) != CUFFT_SUCCESS ) {
			printf("CUFFT error\n");
		}

		if ( cudaThreadSynchronize() != cudaSuccess ) {
				printf("CUFFT error\n");
			}

		//	if ( cufftExecC2C ( plan , d_data , d_data , CUFFT_INVERSE ) != CUFFT_SUCCESS ) {
		//		printf("error\n");
		//	}

		cudaMemcpy(data, d_data, BATCH*ARRAY_SIZE*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

		if ( cudaThreadSynchronize() != cudaSuccess ) {
					printf("error\n");
				}

		double endcufft = getTimeMS();

		if(ARRAY_SIZE==16){

			printf("CUFFT:\n\n");

			for(int  i = 0 ; i < 16 ; i++ ) {
				printf("Elem(%d) = (%f,%f);\n",i,data[i].x,data[i].y);
			}
		}

		printf("(V) GAMA done, in %.4f miliseconds and with %d errors!\n",((end5-start5)+(G01-G00)+(end1-start1)+(end3-start3)+(end4-start4)+(end5-start5)),errors);
		printf("(V) FFTW done, in %.4f miliseconds!\n",(endfftw2-startfftw2) + (endfftw1-startfftw1) );
		printf("(V) CUFFT done, in %.4f miliseconds!\n",(endcufft-startcufft)  );


		delete rs;

}

#endif /* FFT_MAIN_H_ */
