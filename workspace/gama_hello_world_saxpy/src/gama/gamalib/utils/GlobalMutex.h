/*
 * GlobalMutex.h
 *
 *  Created on: May 12, 2011
 *      Author: jbarbosa
 *
 *
 *
 *      Implementation based on Eisenberg & McGuire algorithm
 *
 */

#ifndef GLOBALMUTEX_H_
#define GLOBALMUTEX_H_



typedef unsigned char LOCK_STATE;

const char LOCK_IDLE = 0;
const char LOCK_WAITING = 1;
const char LOCK_ACTIVE = 2;
const char LOCK_ALIGN = 3;

template <unsigned int DEVICES> class GlobalMutex {

public:
	LOCK_STATE flags[DEVICES];

	volatile unsigned int turn;




	__DEVICE__ GlobalMutex() : turn(0) {
		for (unsigned int i = 0; i < DEVICES; i++)
			flags[i] = LOCK_IDLE;
	}

	__DEVICE__ ~GlobalMutex() {
	}

	__DEVICE__
	bool Acquire(unsigned int threadID) {
		unsigned int index = turn;
		/* if there were no other active processes, AND if we have the turn
		 or else whoever has it is idle, then proceed.  Otherwise, repeat
		 the whole sequence. */
		unsigned int count = 10*DEVICES;

		flags[threadID] = LOCK_WAITING;

		while ( ! ((index == DEVICES) && ( turn==threadID || flags[turn] == LOCK_IDLE) ) ) {

			/* announce that we need the resource */
			flags[threadID] = LOCK_WAITING;

			/* scan processes from the one with the turn up to ourselves. */
			/* repeat if necessary until the scan finds all processes idle */
			index = turn;

			while (index != threadID) {
				if (flags[index] != LOCK_IDLE)
					index = turn;
				else
					index = (index + 1) % DEVICES;

				if (index == turn) {
					count--;
					if (count == 0){
						flags[threadID] = LOCK_IDLE;
						return false;
					}
				}
			}

			/* now tentatively claim the resource */
			flags[threadID] = LOCK_ACTIVE;

			/* find the first active process besides ourselves, if any */
			index = 0;

			while ((index < DEVICES) && ((index == threadID) || (flags[index] != LOCK_ACTIVE))) {
				index = index + 1;
			}
		}



		/* Start of CRITICAL SECTION */

		/* claim the turn and proceed */
		turn = threadID;
		//#ifdef __CUDACC__
		//		}
		//#endif
		//printf("Out of lock (%d, %d , %d, %d) \n",threadID,turn, index, flags[threadID]);
		/* Critical Section Code of the Process */

		return true;

	}

	__DEVICE__
	bool Release(int threadID) {
		/* End of CRITICAL SECTION */
		//#ifdef __CUDACC__
		//		if((threadIdx.x % 32) == 0) {
		//#endif
		/* find a process which is not IDLE */
		/* (if there are no others, we will find ourselves) */
		int index_old = turn;

		int index = (turn + 1) % DEVICES;

		while (flags[index] == LOCK_IDLE && index != index_old) {
			index = (index + 1) % DEVICES;
		}

		/* give the turn to someone that needs it, or keep it */

		turn = index;

		/* we're finished now */
		flags[threadID] = LOCK_IDLE;
		//printf("Release out (%d, %d , %d) \n",threadID,turn, flags[threadID]);
		/* REMAINDER Section */
		//#ifdef __CUDACC__
		//		}
		//#endif
		return true;
	}

};

#endif /* GLOBALMUTEX_H_ */
