#ifndef _STARPU_WRAPPER_TYPES_
#define _STARPU_WRAPPER_TYPES_

namespace starpu {

  // cpu function pointer
  typedef void (*cpu_func_t)(void* buffers[], void* cl_args);

  // cuda function pointer

  // codelet where clauses
  #define STARPU_BOTH (STARPU_CPU|STARPU_CUDA)

}


#endif // _STARPU_WRAPPER_TYPES
