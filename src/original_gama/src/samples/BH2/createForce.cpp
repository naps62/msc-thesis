/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <config/common.h>
#include <gamalib/gamalib.h>

#if (SAMPLE==5)
#include "createForce.h"

BHForce* createForce(
		smartPtr<float> m,
		smartPtr<float> px, smartPtr<float> py, smartPtr<float> pz,
		smartPtr<float> vx, smartPtr<float> vy, smartPtr<float> vz,
		smartPtr<float> ax, smartPtr<float> ay, smartPtr<float> az
		,smartPtr<int> ch, smartPtr<int> st
		,unsigned long _lower, unsigned long _upper, unsigned long step,float* dq, unsigned long nnodesd,
		int maxd, float grad
		,unsigned int* c
		)
{
	return new BHForce(
	m,
	px,py,pz,
	vx,vy,vz,
	ax,ay,az,
	ch,st,
	_lower,_upper,(unsigned long)step,(float*)dq,nnodesd,maxd,grad,c);
}

void deleteForce(BHForce* fc) {
	delete fc;
}
#endif
