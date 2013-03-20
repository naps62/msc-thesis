/*
 * worker.h
 *
 *  Created on: May 2, 2012
 *      Author: amariano
 */

#ifndef Proxy_H_
#define Proxy_H_

#include "PerformanceModel.h"
#include <signal.h>

#include <config/system.cfg.h>
#include <config/workqueue.cfg.h>

#include <gamalib/workqueues/Workqueue.h>

typedef struct gpu_assigns{

	unsigned id;
	unsigned long assigns;
	unsigned long executions;

}*GPU_ASSIGNS;

typedef struct rob_ctrl{

	int worktype;
	unsigned long dices;
	unsigned long asgns;

}*ROB_CTRL;

class Proxy {
public:

	pthread_t thread;
	pthread_attr_t attr;
	void *t_status;

	pthread_mutex_t mutex1;
	pthread_cond_t cond1;

	FIFOLockFreeQueue<work*>* _ROB;

	Workqueue<work, DEVICE_QUEUE_SIZE, CPU_X86>* _queues[TOTAL_DEVICES];

	PerformanceModel* _pModels[TOTAL_DEVICES];

	volatile bool globalEmpty;

	int assignNumber;
	int kill;

	unsigned long* pointers[TOTAL_DEVICES];

	GPU_ASSIGNS* devices;
	unsigned gpu_number;

	ROB_CTRL * tasks;
	int currentCTRL, lengthCTRL, allocated;

	float _execTimesW[TOTAL_DEVICES];

	int last_worktype;

public:

	Proxy() {
	}

	Proxy(FIFOLockFreeQueue<work*>* ROB,
			Workqueue<work, DEVICE_QUEUE_SIZE, CPU_X86>* queues[],
			PerformanceModel* pm[], unsigned gpus, unsigned IDsGPUs[] ) {

		_ROB = ROB;

		for(unsigned dev=0; dev < TOTAL_DEVICES; dev++){
			_queues[dev] = queues[dev];
			_pModels[dev] = pm[dev];
		}

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		assignNumber = 0;
		kill = 0;

		globalEmpty = true;

		for( int dev = 0; dev < TOTAL_DEVICES ; dev++ ){
			pointers[dev] = (unsigned long*)malloc(3*NKERNELS*(sizeof(unsigned long)));

			for( int ker = 0 ; ker < NKERNELS ; ker++){

				pointers[dev][ker*3] = 0;
				pointers[dev][ker*3+1] = 0;
				pointers[dev][ker*3+2] = 0;

			}

		}

		for( int dev = 0; dev < TOTAL_DEVICES ; dev++ )
			_execTimesW[dev] = 100.0;

		gpu_number = gpus;

		initDevicesAssigns(IDsGPUs);

		initROBControl();

		pthread_cond_init (&cond1,NULL);
		pthread_mutex_init(&mutex1,NULL);

	}

	void launch() {

		pthread_create(&thread, NULL, Proxy::run, (void*) this);

	}

	void assign(){

		//printf("WORKER: I am going to wake the Proxy thread! \n");
		int er = pthread_cond_signal(&cond1);
		//printf("WORKER: Done, with message = %d! \n",er);

	}

	static void* run (void* params){

		Proxy* a = reinterpret_cast<Proxy*>(params);

		a->heftAssign(params);

		::pthread_exit(NULL);
	}

	bool isGlEmpty(){

		return globalEmpty;

	}

	bool setGLEmpty(bool p, int workID){

		globalEmpty = p;

		for( int dev = 0; dev < TOTAL_DEVICES ; dev++ ){
			for( int ker = 0 ; ker < NKERNELS ; ker++){
				if( pointers[dev][ker*3]==0 ){
					pointers[dev][ker*3] = workID;
					break;
				}
				else if( pointers[dev][ker*3]==workID ) break;
			}
		}
		return true;
	}

	void addExecution(int deviceID, int workID){

		for( int ker = 0 ; ker < NKERNELS ; ker++ )
			if( pointers[deviceID][ker*3] == workID){
				pointers[deviceID][ker*3+2] += 1;
				break;
			}

	}

	void addAssigning(int deviceID, int workID, int amount){

		for( int ker = 0 ; ker < NKERNELS ; ker++ )
			if( pointers[deviceID][ker*3] == workID){
				pointers[deviceID][ker*3+1] += amount;
				break;
			}

		//setATasksROBWT(amount,workID);

	}

	void printHistory(int deviceID){

		for( int ker = 0 ; ker < NKERNELS ; ker++ )

			printf("GAMA assigned %lu tasks of worktype %lu to the device %lu. It executed %lu.\n",(unsigned long)pointers[deviceID][ker*3+1],(unsigned long)pointers[deviceID][ker*3],(unsigned long)deviceID,(unsigned long)pointers[deviceID][ker*3+2]);

	}

	void initDevicesAssigns( unsigned IDsGPUs[] ){

		devices = (GPU_ASSIGNS*)malloc(gpu_number*(sizeof(unsigned long)));

		for( int gpu = 0 ; gpu < gpu_number ; gpu++ ){
			devices[gpu] = (GPU_ASSIGNS)malloc(sizeof(struct gpu_assigns));
			devices[gpu]->assigns = 0;
			devices[gpu]->executions = 0;
			devices[gpu]->id = IDsGPUs[gpu];
		}

	}

	void addDeviceAssign( unsigned id, unsigned amount ){

		for( int gpu = 0 ; gpu < gpu_number ; gpu++ )
			if( devices[gpu]->id == id )
				devices[gpu]->assigns+= amount;

	}

	void addDeviceExecution( unsigned id ){

		for( int gpu = 0 ; gpu < gpu_number ; gpu++ )
			if( devices[gpu]->id == id )
				devices[gpu]->executions++;

	}

	int queryDriverQueueSize( unsigned id ){

		for( int gpu = 0 ; gpu < gpu_number ; gpu++ )
			if( devices[gpu]->id == id )
				return devices[gpu]->assigns - devices[gpu]->executions;

		return 0;

	}

	void initROBControl(){

		tasks = (ROB_CTRL*)malloc(2*NKERNELS*sizeof(struct rob_ctrl));

		currentCTRL = 0;
		lengthCTRL = 0;
		allocated = 2*NKERNELS;

	}

	void setDTasksROBWT( unsigned long ds, int worktype ){

		if( lengthCTRL == allocated ){

			ROB_CTRL *realloced_tasks;
			realloced_tasks = (ROB_CTRL*)realloc(tasks,4*allocated*sizeof(struct rob_ctrl));
			tasks = realloced_tasks;

			allocated = allocated*4;

		}

		tasks[lengthCTRL] = (ROB_CTRL)malloc(sizeof(struct rob_ctrl));

		tasks[lengthCTRL]->dices = ds;
		tasks[lengthCTRL]->asgns = 0;
		tasks[lengthCTRL++]->worktype = worktype;

	}

	void setATasksROBWT( unsigned long as, int worktype ){

		tasks[currentCTRL]->asgns += as;

		unsigned val = tasks[currentCTRL]->asgns - tasks[currentCTRL]->dices;

		if( val == 0 && (tasks[currentCTRL]->dices != 0) )
			currentCTRL++;

	}

	unsigned getTasksROBWT( int* worktype ){

		*worktype = tasks[currentCTRL]->worktype;

		return tasks[currentCTRL]->dices - tasks[currentCTRL]->asgns;

	}

	int getCurrentWorktype(  ){

		return tasks[currentCTRL]->worktype;

	}

	void* heftAssign (void* params);

};

#endif /* Proxy_H_ */
