#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H

#include <starpu_cuda.h>

#ifdef __cplusplus
extern "C"
#endif
void cuda_kernel(void *buffers[], void *args);

#endif // _CUDA_KERNEL_
