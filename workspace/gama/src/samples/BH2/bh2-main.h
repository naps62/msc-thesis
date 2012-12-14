#include "bh-martin.h"



int main(int argc, char* argv[]) {

	RuntimeScheduler *rs = new RuntimeScheduler();


	unsigned int nnodes = NBODIES * 2;

	if (nnodes < 1024*2) nnodes = 1024*2;
	while ((nnodes & (32-1)) != 0) nnodes++;
	nnodes--;


	smartPtr<int> childl( sizeof(int) * (nnodes+1) * 8);
	smartPtr<float> massl(sizeof(float) * (nnodes+1));
	smartPtr<float> posxl(sizeof(float) * (nnodes+1));
	smartPtr<float> posyl(sizeof(float) * (nnodes+1));
	smartPtr<float> poszl(sizeof(float) * (nnodes+1));



	burtcher(argc,argv,rs,NBODIES,TIMESTEPS,massl.getPtr(),posxl.getPtr(),posyl.getPtr(),poszl.getPtr(),childl.getPtr());


}
