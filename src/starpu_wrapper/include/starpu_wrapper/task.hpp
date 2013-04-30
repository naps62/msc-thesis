#ifndef _STARPU_WRAPPER_TASK_
#define _STARPU_WRAPPER_TASK_

#include <starpu_wrapper/codelet.hpp>

namespace starpu {

  struct task {

    task() {
      t = starpu_task_create();
    }

    task(codelet cl) {
      t = starpu_task_create();
      set_codelet(cl);
    }

    task& set_codelet(codelet cl) {
      t->cl = cl.ptr();
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
  };
}

#endif // _STARPU_WRAPPER_TASK_
