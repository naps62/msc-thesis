
#ifndef CREATE_FORCE_H
#define CREATE_FORCE_H

BHForce* createForce(
		smartPtr<float> m,
		smartPtr<float> px, smartPtr<float> py,smartPtr<float> pz,
		smartPtr<float> vx, smartPtr<float> vy, smartPtr<float> vz,
		smartPtr<float> ax, smartPtr<float> ay, smartPtr<float> az
		,smartPtr<int> ch, smartPtr<int> st
		,unsigned long _lower, unsigned long _upper, unsigned long step,float* dq, unsigned long nnodesd,
		int maxd, float grad
		, unsigned int* c
		);
void deleteForce(BHForce* fc);

#endif
