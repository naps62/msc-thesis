#ifndef _STARPU_WRAPPER_CODELET_
#define _STARPU_WRAPPER_CODELET_

#include <starpu_wrapper/types.hpp>

extern "C" {
  struct starpu_codelet __default_cl = {};
}

namespace starpu {

  struct codelet {

    codelet(uint32_t where = 0, cpu_func_t cpu_func = NULL) {
      n_cpu_funcs = n_cuda_funcs = 0;
      init_codelet(where, cpu_func);
    }

    codelet& add_where(uint32_t new_where) {
      cl.where |= new_where;
      return *this;
    }

    codelet& cpu_func(cpu_func_t func) {
      cl.cpu_funcs[n_cpu_funcs++] = func;
      return *this;
    }

    starpu_codelet* ptr() {
      return &cl;
    }

  private:
    starpu_codelet cl;
    unsigned int n_cpu_funcs;
    unsigned int n_cuda_funcs;

    void init_codelet(uint32_t where, cpu_func_t cpu_func) {
      memcpy(&cl, &__default_cl, sizeof(starpu_codelet));
      cl.cpu_funcs[0] = NULL;
      cl.cuda_funcs[0] = NULL;
      this->add_where(where);
      if (cpu_func)
        this->cpu_func(cpu_func);
    }
  };
}

#endif // _STARPU_WRAPPER_CODELET_
