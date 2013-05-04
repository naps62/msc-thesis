#ifndef _STARPU_WRAPPER_TASK_
#define _STARPU_WRAPPER_TASK_

#include <starpu_wrapper/types.hpp>
#include <starpu_wrapper/codelet.hpp>

namespace starpu {

  struct task {

    task() {
      n_handles = 0;
      t = starpu_task_create();
    }

    task(codelet cl) {
      n_handles = 0;
      t = starpu_task_create();
      set_codelet(cl);
    }

    task& set_codelet(codelet cl) {
      t->cl = cl.ptr();
      return *this;
    }

    task& handle(starpu_data_handle_t handle) {
      t->handles[n_handles++] = handle;
      return *this;
    }

    template<typename T>
    task& handle(starpu::vector<T> vec) {
      t->handles[n_handles++] = vec.handle();
      return *this;
    }

    template<typename ArgClass>
    task& arg(ArgClass* arg) {
      t->cl_arg = arg;
      t->cl_arg_size = sizeof(ArgClass);
      return *this;
    }

    task& callback(callback_func_t callback, void* arg = NULL) {
      t->callback_func = callback;
      t->callback_arg  = arg;
      return *this;
    }

    task& set_sync(bool sync = true) {
      t->synchronous = sync;
      return *this;
    }

    starpu_task* ptr() {
      return t;
    }


  private:
    starpu_task* t;
    unsigned int n_handles;
  };
}

#endif // _STARPU_WRAPPER_TASK_
