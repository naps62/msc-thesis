#include "proxy.h"

#include <lapackpp/lapackpp.h>

void* Proxy::heftAssign (void* params){

		Proxy* a = reinterpret_cast<Proxy*>(params);

		printf("(Sys) Dynamic scheduler activated!\n");

		//Current worktype under scheduling:
		int worktype;

		while(true){

			//printf("Proxy THREAD: I am going to sleep now!\n");

			pthread_cond_wait(&a->cond1, &a->mutex1);

			//printf("Proxy THREAD: I got a signal...\n");

			//printf("Proxy THREAD: I am awake, so I'll assign!\n");


			while(!a->_ROB->trylock());

			//printf("got lock\n");

			if(!a->_ROB->isEmpty()){

				//printf("não está vazia!\n");

				//if(_ROB->isEmpty()) a->globalEmpty = true;
				//else a->globalEmpty = false;

//	 =============================================================================================================== HEFT:

				//Current worktype under scheduling:
				//int worktype;

				//Proxy might not be able to find SCHEDULER_CHUNK tasks belonging to the worktype under schedule: this is the maximum loads to schedule:
				int maxLoads = a->getTasksROBWT(&worktype);

				//Debug:
				//printf("Proxy THREAD: Faltam %d do current com worktype %d!\n",maxLoads,worktype );

				//Array with execution times of task of WORKTYPE 'w' for each device:
				float execTimesW[TOTAL_DEVICES];

				unsigned devsWthET[TOTAL_DEVICES];

				memcpy ( execTimesW, _execTimesW, TOTAL_DEVICES*4 );

				int devsWthETLength = 0;

				float execTimeW, biggest = -1;

				for( unsigned dev=0; dev < TOTAL_DEVICES; dev++ ){
					execTimeW = a->_pModels[dev]->searchTaskSize(worktype,128);
					if(execTimeW!=-1 && execTimeW!=0.f){
						execTimesW[dev] = execTimeW;
						if(execTimeW > biggest){
							biggest = execTimeW;
						}
					}
					else{
						devsWthET[devsWthETLength++] = dev;
					}
				}

				unsigned ptr;

				if( biggest!= -1.f && biggest != 0.f )
					for( int dev=0; dev < devsWthETLength; dev++ ){
						ptr = devsWthET[dev];
						execTimesW[ptr] = (biggest*((a->assignNumber)*2));
						//printf("%d mudado\n",ptr);
					}

//				printf("Biggest = %f\n",biggest );
//
//				for( unsigned dev=0; dev < TOTAL_DEVICES; dev++ )
//					printf("Exec. Time dev %d = %f\n",dev,execTimesW[dev]);
//
//				printf("...\n");

				//Estimated delay time of each queue of each device:
				float times[TOTAL_DEVICES];

				memset( times, 0, TOTAL_DEVICES*4 );

				float aux_value;
				int aux_wt;

				//Calculus of each queue's time:
				for( unsigned dev=0; dev < TOTAL_DEVICES; dev++ )
					for( int ker=0 ; ker < NKERNELS ; ker++ ){

						aux_wt =  a->pointers[dev][ker*3];

						if( aux_wt == worktype ) aux_value = execTimesW[dev];
						else aux_value = a->_pModels[dev]->searchTaskSize(aux_wt,128);

						if( aux_value>0 ){
							times[dev] += ((a->pointers[dev][ker*3+1] - a->pointers[dev][ker*3+2]) * aux_value) + (queryDriverQueueSize(dev)*aux_value);
							//TODO:
							//printf("Query dev %d = %d\n",dev,(queryDriverQueueSize(dev)-1));
//							if(dev==2) printf("Query dev %d = %f\n",dev,(queryDriverQueueSize(dev)*aux_value));
						}

						else{
							//TODO:
							times[dev] += ((a->pointers[dev][ker*3+1] - a->pointers[dev][ker*3+2]) + queryDriverQueueSize(dev));
//							if(dev==2) printf("Query dev %d = %f\n",dev,(queryDriverQueueSize(dev)*aux_value));
						}
					}

//				printf("\n");
//				for( unsigned dev=0; dev < TOTAL_DEVICES; dev++ )
//					printf("Queue dev %d = %f\n",dev,times[dev]);
//				printf("\n");

				/* HEFT to assign x + y tasks of work_type W */

				double mAA[TOTAL_DEVICES*TOTAL_DEVICES];// = {0};

				memset ( mAA, 0, (TOTAL_DEVICES*TOTAL_DEVICES-1)*8 );

				//Last matrix's row:
				for( unsigned j=0; j < TOTAL_DEVICES; j++ )
					mAA[(TOTAL_DEVICES*(TOTAL_DEVICES-1))+j] = 1.f;

			    LaGenMatDouble mA(mAA,TOTAL_DEVICES,TOTAL_DEVICES,true);
			    LaVectorDouble mX(TOTAL_DEVICES);
			    LaVectorDouble mB(TOTAL_DEVICES);

			    for( unsigned i=0; i < TOTAL_DEVICES-1; i++ )
			    	for( unsigned j=0; j < 2; j++ ){
			    		mA(i,i+j) = execTimesW[j+i];
			    		if(j==1) mA(i,i+j) = -(execTimesW[j+i]);
			    	}

			    //B values:
			    for( int dev=0; dev < (TOTAL_DEVICES-1); dev++ ){
			    	mB(dev) = times[dev+1] - times[dev];
			    	//printf("mB(%d) = times[%d](%.1f) - times[%d](%.1f);\n",dev,dev+1,times[dev+1],dev,times[dev]);
			    }

			    //Last B value (x + y + ... + n = SCHEDULER_CHUNK)
			    mB(TOTAL_DEVICES-1) = SCHEDULER_CHUNK;

			    // Linear equation system print:
//			    for( unsigned i=0; i < TOTAL_DEVICES; i++ ){
//			    	for( unsigned j=0; j < TOTAL_DEVICES+1; j++ ){
//			    		if(j==TOTAL_DEVICES) printf("%f ",mB(i));
//			    		else printf("%f ",mA(i,j));
//			    	}
//			    	printf("\n");
//			    }
			    //getchar();

			    //TODO:
			    //Verificar se o sistema é sempre possível:
			    LaLinearSolve(mA,mX,mB);

				float total = 0;

				int totalPDevices = 0, PDevices[TOTAL_DEVICES];

				for( unsigned int dev = 0 ; dev < TOTAL_DEVICES ; dev++ ){

					if((int)mX(dev)>0){
						total+= mX(dev);
						PDevices[totalPDevices++] = dev;
						//printf("dev %d anotado com %f units!\n",dev,mX(dev));
					}
					else mX(dev) = 0.0;

				}

				int reg;

				for( unsigned int dev = 0 ; dev < totalPDevices ; dev++ ){

					reg = PDevices[dev];

					mX(reg) /= total;
					mX(reg) *= SCHEDULER_CHUNK;

				}

//	 =============================================================================================================== HEFT;

//				for( unsigned dev=0; dev < TOTAL_DEVICES; dev++ )
//					printf("Device %d com %d tarefas;\n",dev,((int)floor(mX(dev)+0.5)));
//
//				printf("...\n");

				int iterations, tAssignings=0;

				work* wu;

				for( unsigned int dev = 0 ; dev < totalPDevices && (tAssignings < maxLoads) ; dev++ ){

					reg = PDevices[dev];

					for( iterations = 0; (iterations < ((int)floor(mX(reg)+0.5))) && (!a->_ROB->isEmpty()) && (!a->_queues[reg]->isFull()) && (tAssignings < maxLoads) ; ){

						if(a->_ROB->dequeue(wu)){

							a->_queues[reg]->enqueue(wu);
							iterations++;
							tAssignings++;

						}

					}

					if(iterations){
						a->addAssigning(reg,worktype,iterations);
					}

					//printf("loads[%d] = %d; \n",reg,it);

				}

				a->setATasksROBWT(tAssignings,worktype);

//				printf(" ---- \n");

//				if(asss){
//					printf("Proxy THREAD: I've assigned %d works and there are %d remaining in the ROB!\n",asss,a->_ROB->size() );
//				}

				//getchar();

				if(a->_ROB->isEmpty()) a->globalEmpty = true;
				else a->globalEmpty = false;

				a->assignNumber++;

			}

			a->_ROB->release();

			//TODO:
			//if( a->kill ) pthread_exit(NULL);

		}

}





























//testes:

//void* Proxy::heftAssign (void* params){
//
//		Proxy* a = reinterpret_cast<Proxy*>(params);
//
//		printf("(Sys) Dynamic scheduler activated!\n");
//
//		int loads[TOTAL_DEVICES];
//
//		while(true){
//
//			//printf("Proxy THREAD: I am going to sleep now!\n");
//
//			pthread_cond_wait(&a->cond1, &a->mutex1);
//
//			//printf("Proxy THREAD: I got a signal...\n");
//
//			//printf("Proxy THREAD: I am awake, so I'll assign!\n");
//
//
//			while(!a->_ROB->trylock());
//
//			//printf("got lock\n");
//
//			if(!a->_ROB->isEmpty()){
//
//				//printf("ass number = %d\n",a->assignNumber);
//
////	 =============================================================================================================== HEFT:
//
//				for( int i=0; i < TOTAL_DEVICES; i++)
//					loads[i] = 0;
//
//				if(a->assignNumber==0){
//
//					loads[0] = 2;
//					loads[1] = 2;
//					loads[2] = 2;
//					loads[3] = 2;
//					loads[4] = 2;
//
//				}
//				else if(a->assignNumber==1){
//
//					loads[0] = 1;
//					loads[1] = 1;
//					loads[2] = 3;
//					loads[3] = 1;
//					loads[4] = 1;
//
//				}
//				else if(a->assignNumber==2){
//
//					loads[1] = 2;
//					loads[2] = 6;
//				}
//				else if(a->assignNumber==3){
//
//					loads[2] = 4;
//					loads[3] = 4;
//
//				}
//				else if(a->assignNumber==4){
//
//					loads[0] = 2;
//					loads[1] = 2;
//					loads[2] = 3;
//					loads[4] = 1;
//
//				}
//				else if(a->assignNumber==5){
//
//					loads[1] = 2;
//					loads[2] = 3;
//					loads[3] = 1;
//					loads[4] = 2;
//
//				}
//				else if(a->assignNumber==6){
//
//					loads[0] = 2;
//					loads[2] = 4;
//					loads[3] = 2;
//
//				}
//				else if(a->assignNumber==7){
//
//					loads[1] = 4;
//					loads[2] = 4;
//
//
//				}
//				else if(a->assignNumber==8){
//
//					loads[0] = 1;
//					loads[2] = 2;
//					loads[3] = 2;
//					loads[4] = 2;
//
//				}
//				else if(a->assignNumber==9){
//
//					loads[0] = 2;
//					loads[1] = 2;
//					loads[2] = 3;
//					loads[3] = 1;
//
//				}
//				else if(a->assignNumber==10){
//
//					loads[2] = 3;
//					loads[3] = 2;
//					loads[4] = 2;
//
//				}
//				else if(a->assignNumber==11){
//
//					loads[1] = 3;
//					loads[2] = 4;
//					loads[3] = 2;
//				}
//				else if(a->assignNumber==12){
//
//					loads[0] = 2;
//					loads[2] = 4;
//					loads[4] = 3;
//
//				}
//				else if(a->assignNumber==13){
//
//					loads[0] = 1;
//					loads[1] = 2;
//					loads[2] = 2;
//					loads[3] = 3;
//
//				}
//				else if(a->assignNumber==14){
//
//					loads[1] = 3;
//					loads[2] = 4;
//					loads[3] = 2;
//
//				}
//				else if(a->assignNumber==15){
//
//					loads[0] = 2;
//					loads[2] = 3;
//					loads[4] = 1;
//				}
//				else if(a->assignNumber==16){
//
//					printf("%d\n",a->_ROB->size());
//				}
//
//				int it, asss=0;
//
//				work* wu;
//
//				for( unsigned int dev = 0 ; dev < TOTAL_DEVICES  ; dev++ ){
//
//					for( it = 0; (it < loads[dev]) && (!a->_ROB->isEmpty()) && (!a->_queues[dev]->isFull()) ; ){
//
//						if(a->_ROB->dequeue(wu)){
//
//							a->_queues[dev]->enqueue(wu);
//							it++;
//							asss++;
//
//						}
//
//					}
//
////					if(it){
////						a->addAssigning(reg,worktype,it);
////					}
//
//					if(it){
//						printf("dev %d = %d tasks \n",dev,it);
//					}
//				}
//
////				a->setATasksROBWT(asss,worktype);
//
//				if(a->assignNumber<20) printf(" ---- \n");
//
////				if(asss){
////					printf("Proxy THREAD: I've assigned %d works and there are %d remaining in the ROB!\n",asss,a->_ROB->size() );
////				}
//
//				//getchar();
//
//				if(a->_ROB->isEmpty()) a->globalEmpty = true;
//				else a->globalEmpty = false;
//
//				//a->CHUNK *= 2;
//				a->assignNumber++;
//
//			}
//
//			a->_ROB->release();
//
//			//if( a->kill ) pthread_exit(NULL);
//
//		}
//
//}
