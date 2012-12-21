

#ifndef __VTABLE_X86_H_
#define __VTABLE_X86_H_


#include <config/work.cfg.h>
#include <gamalib/work/work.h>

/*
 *
 * Include work description files
 *
 */


typedef work pWork;

typedef void (work::*executeFN)(void);
typedef List<work*>* (work::*diceFN)(unsigned int &number);
typedef std::vector<pointerInfo>* (work::*toCacheRFN)();
typedef std::vector<pointerInfo>* (work::*toCacheWFN)();
//typedef bool (work::*cacheFN)(CACHE* _cache);
//typedef bool (work::*evictFN)(void);
//typedef void (work::*destructorFN)(void);

#include <gamaconf/vtable.h>

//#ifndef __CUDACC__
//const executeFN WORK_CPU_TABLE[WORK_TOTAL] = {
//	&work::execute<CPU_X86>,
//	#error add cpu work table
//	// (executeFN)&saxpy::execute<CPU_X86>,
//};
//#endif
//
//const diceFN DICE_CPU_TABLE[WORK_TOTAL] = {
//	&work::dice<CPU_X86>,
//	#error add cpu dice table
//	// (diceFN)&saxpy::dice<CPU_X86>,
//};
//
//const toCacheRFN TOCACHER_CPU_TABLE[WORK_TOTAL] = {
//	&work::toCacheR,
//	#error add cpu toCacheR cpu table
//	// (toCacheRFN)&saxpy::toCacheR,
//};
//
//const toCacheRFN TOCACHEW_CPU_TABLE[WORK_TOTAL] = {
//	&work::toCacheR,
//	#error add toCacheW cpu table
//	// (toCacheWFN)&saxpy::toCacheW,
//};


#endif
