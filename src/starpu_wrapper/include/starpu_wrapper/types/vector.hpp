#ifndef _STARPU_WRAPPER_VECTOR_
#define _STARPU_WRAPPER_VECTOR_

#include <vector>
#include <starpu.h>

namespace starpu {

  template<typename T>
  struct vector : public std::vector<T> {

    vector() : std::vector<T>() { }
    vector(unsigned n) : std::vector<T>(n) { }

    void register_data() {
      starpu_vector_data_register(&_handle, 0, (uintptr_t)&((*this)[0]), this->size(), sizeof(T));
    }

    void unregister() {
      starpu_data_unregister(_handle);
    }

    void acquire() {
      starpu_data_acquire(_handle, STARPU_R);
    }

    starpu_data_handle_t handle() {
      return _handle;
    }

  private:
    starpu_data_handle_t _handle;
  };
}

#endif // _STARPU_WRAPPER_TASK_
