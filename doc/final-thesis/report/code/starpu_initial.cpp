void render() {
  starpu_data_register(seeds, ...)
  starpu_data_register(eye_paths, ...)
  starpu_data_register(hit_points, ...)

  starpu_insert_task(codelets::init_seeds, seeds);

  int iteration = 0;
  while(iteration < config.max_iters) {
    starpu_insert_task(codelets::generate_eye_paths, eye_paths, seeds);
    starpu_insert_task(codelets::advance_eye_paths,  eye_paths, seeds, hit_points);

    ... // all other tasks for this iteration

    iteration++;
  }

  starpu_data_unregister(seeds);
  starpu_data_unregister(eye_paths);
  starpu_data_unregister(hit_points);
}
