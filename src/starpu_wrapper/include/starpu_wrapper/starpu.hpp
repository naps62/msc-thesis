#ifndef _STARPU_WRAPPER_STARPU_
#define _STARPU_WRAPPER_STARPU_

#include <starpu.h>
#include <starpu_wrapper/task.hpp>

namespace starpu {
  void init() {
    starpu_init(NULL);
  }

  void shutdown() {
    starpu_shutdown();
  }

  void submit(task t) {
    starpu_task_submit(t.ptr());
  }
}

#endif // _STARPU_WRAPPER_STARPU_
