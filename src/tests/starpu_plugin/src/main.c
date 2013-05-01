// Task declaration
static void my_task (int x) __attribute__ ((task));

// Definition of the CPU implementation of 'my_task'
static void my_task (int x) {
 printf("Hello Workd! With x = %d\n", x);
}

int main() {
  #pragma starpu initialize
  // do an asynchronous call to 'my_task'
  my_task(42);
  #pragma starpu wait
  #pragma starpu shutdown
  return 0;
}
