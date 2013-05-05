#ifndef _STARPU_WRAPPER_CODELET_
#define _STARPU_WRAPPER_CODELET_

#include <starpu_wrapper/types.hpp>

namespace starpu {

  struct codelet {

    codelet(uint32_t where = 0, cpu_func_t cpu_func = NULL, cuda_func_t cuda_func = NULL) {
      n_cpu_funcs = n_cuda_funcs = n_modes = 0;
      starpu_codelet_init(&cl);
      this->init_codelet(where, cpu_func, cuda_func);
    }

    codelet& add_where(uint32_t new_where) {
      cl.where |= new_where;
      return *this;
    }

    codelet& cpu_func(cpu_func_t func) {
      cl.cpu_funcs[n_cpu_funcs++] = func;
      return *this;
    }

    codelet& cuda_func(cuda_func_t func) {
      cl.cuda_funcs[n_cuda_funcs++] = func;
      return *this;
    }

    codelet& buffer(starpu_access_mode mode) {
      cl.modes[n_modes++] = mode;
      cl.nbuffers = n_modes;
      return *this;
    }

    starpu_codelet* ptr() {
      return &cl;
    }

  private:
    starpu_codelet cl;
    unsigned int n_cpu_funcs;
    unsigned int n_cuda_funcs;
    unsigned int n_modes;

    void init_codelet(uint32_t where, cpu_func_t cpu_func, cuda_func_t cuda_func) {
      cl.cpu_funcs[0] = NULL;
      cl.cuda_funcs[0] = NULL;
      this->add_where(where);
      if (cpu_func) this->cpu_func(cpu_func);
      if (cuda_func) this->cuda_func(cuda_func);
    }
  };
}

#endif // _STARPU_WRAPPER_CODELET_
