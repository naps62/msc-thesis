/*
 * PerformanceModel.h
 *
 *  Created on: Apr 25, 2012
 *      Author: amariano
 */

#ifndef PERFORMANCEMODEL_H_
#define PERFORMANCEMODEL_H_

#define MAX_TASKS 20
#define MAX_EXECS 5
#define MAX_TRIALS 5

#include <gamalib/workqueues/FifoLockFreeQueue.h>
#include <gamalib/work/work.h>
#include <iostream>
#include <stdlib.h>

//Element of the second HashTable - TIME
//
typedef struct time{
	int size;
	float trials[MAX_TRIALS];
	float exec_time;
	float totalTrials;
	int measures;
	//int pointer;
	struct time *next;
}*TIME;
//

//Element of the initial HashTable - TASK
//
typedef struct task{
	int ID;
	struct task *next;
	TIME HashTable2[MAX_EXECS];

}*TASK;
//

class PerformanceModel;

typedef struct arguments{
	float worktodo;
	FIFOLockFreeQueue<work*>* rob;
	//PerformanceModel:
	PerformanceModel* pf;
}*ARGZ;

typedef TASK* HashTable;
typedef TIME* HashTable2;

class PerformanceModel {
private:
	int ID;
	int device;
	TASK pf[MAX_TASKS];

public:
	PerformanceModel( ){

		initializeHashT();

	}

	//This function initializes the HashTable: if is there any info, the function makes the HashTable return's to the initialized state;
	void initializeHashT(){

	   for(int i=0;i<MAX_TASKS;i++){
		   pf[i] = NULL;
	   }

	}

	//This function initializes the HashTable2: if is there any info, the function makes the HashTable return's to the initialized state;
	void initializeHashT2(HashTable2 a){

	   for(int i=0;i<MAX_EXECS;i++){
		   a[i] = NULL;
	   }

	}

	//This function clears the HashTable: if is there any info, the function makes the HashTable return's to the initialized state;
	void clearHashT(HashTable a){

		for(int i=0;i<MAX_TASKS;i++){
			if(pf[i]) pf[i] = NULL;
		}

	}

	//This function checks if two keys given as arguments are the same ones or equals;
	int equalsKeyTask(int a, int b){

	   if(a==b) return 1;

	   return 0; //It returns one for True and zero for false;

	}

	int equalsKeyTime(int a, int b){

		if(a==b) return 1;

		return 0; //It returns one for True and zero for false;
	}


	int hashTask(int id){

	   return (id % MAX_TASKS);

	}

	int hashSize(int size){

	   return (size % MAX_EXECS);

	}

	int putTime(HashTable2 a, int size, float time){

		int hashValue = hashSize(size);

		//There is no colision at all:
		if(!a[hashValue]){

			a[hashValue] = (TIME)malloc(sizeof(struct time));
			a[hashValue]->size = size;
			a[hashValue]->exec_time = time;
			a[hashValue]->measures = 1;
			a[hashValue]->trials[0] = time;
			a[hashValue]->totalTrials = time;
			a[hashValue]->next = NULL;

			return 1;
		}
		//Colision or not in the hashtable yet:
		else{

			//In chain:
			TIME suspect = a[hashValue];

			while(suspect){

				if(suspect->size==size){

					if(suspect->measures<MAX_TRIALS){
						suspect->trials[suspect->measures-1] = time;
						suspect->measures++;
						suspect->totalTrials += time;
					}
					else{
						//suspect->measures = 1;
						suspect->totalTrials -= suspect->trials[0];
						suspect->trials[0] = time;
						suspect->totalTrials += time;

					}

					suspect->exec_time = suspect->totalTrials/suspect->measures;

					return 10; //Addition

				}

				suspect = suspect->next;

			}

			//Not in hashTable:
			TIME n = (TIME)malloc(sizeof(struct time));
			n->size = size;
			n->exec_time = time;
			n->measures = 1;
			n->trials[0] = time;
			a[hashValue]->totalTrials = time;

			n->next = a[hashValue];

			a[hashValue] = n;

			return 1;

		}
	}

	TASK searchTaskOnly(int id){

		int hashValue = hashTask(id);

		if(!pf[hashValue]) return NULL;
		else{

			if(pf[hashValue]->ID==id) return pf[hashValue];
			else{

				TASK c = pf[hashValue]->next;

				while(c!=NULL){

					if(c->ID==id) return c;

					c = c->next;

				}

				return NULL;

			}

		}
	}

	//procura por size apenas (recebe hashtable2 e nome) (devolve sim(a[hashValue]->exec_time) ou não(-1))
	float searchSizeOnly(HashTable2 a, int size){

		int hashValue = hashSize(size);

		if(!a[hashValue]){
			return -1;
		}
		else{

			TIME suspect = a[hashValue];

			while(suspect){

				if(suspect->size==size){
						return suspect->exec_time;
				}

				suspect = suspect->next;
			}

			return -1;

		}
	}

	//This function searches for a task and the respective size at the same time:
	float searchTaskSize(int id, int size){

		int hashValue = hashTask(id);

		//Doesn't exist any element in the hashValue position:
		if(!pf[hashValue]) return -1;
		//1 or more elements in the hashValue position:
		else{

			TASK suspect = pf[hashValue];

			while(suspect){

				if(suspect->ID==id){
					return (searchSizeOnly(suspect->HashTable2,size));
				}

				suspect = suspect->next;
			}

			return -1;

		}
	}

	//This function inserts a new task on the hashtable: returns 1 for sucess, -1 for unsucess;
	//
	int putTaskAndTime(int id, int size, float time){

		int hashValue = hashTask(id);

		//There is no colision at all:
		if(!pf[hashValue]){

			pf[hashValue] = (TASK)malloc(sizeof(struct task));
			pf[hashValue]->ID = id;
			initializeHashT2(pf[hashValue]->HashTable2);
			pf[hashValue]->next = NULL;

			return putTime(pf[hashValue]->HashTable2,size,time);

		}
		//Colision or not in the hashtable yet:
		else{

			//In chain:
			TASK suspect = pf[hashValue];

			while(suspect){

				if(suspect->ID==id){
					return putTime(suspect->HashTable2,size,time);
				}

				suspect = suspect->next;

			}

			//Not in hashTable:
			TASK c = (TASK)malloc(sizeof(struct task));
			c->ID = id;
			initializeHashT2(c->HashTable2);
			c->next = pf[hashValue];

			pf[hashValue] = c;

			return putTime(pf[hashValue]->HashTable2,size,time);

		}
	}

//	int listTimes(HashTable2 a){
//
//		for(int i=0;i<MAX_TASKS;i++){
//			if(a[i])
//				cout << "Exec time (" << i << ") = " << a[i]->exec_time << endl;
//		}
//
//	}

	//lista todos os tempos de uma tarefa
//	int listTimesTask(HashTable a, int id){
//
//		int hashValue = hashTask(id);
//
//		if(!a[hashValue]) return -1;
//		else{
//
//			if(a[hashValue]->ID==id) return listTimes(a[hashValue]->HashTable2);
//			else{
//
//				TASK c = a[hashValue]->next;
//
//				while(c!=NULL){
//
//					if(c->ID==id) return listTimes(c->HashTable2);
//
//					c = c->next;
//
//				}
//
//				return -1;
//
//			}
//
//		}
//
//		return -1;
//
//	}

};

#endif /* PERFORMANCEMODEL_H_ */
