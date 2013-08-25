void render() {
  starpu_data_register(seeds, ...)

  starpu_insert_task(codelets::init_seeds, seeds);

  int iteration = 0;
  while(iteration < config.max_iters) {
    starpu_data_register(eye_paths, ...)
    starpu_insert_task(codelets::generate_eye_paths, eye_paths, seeds);
    starpu_data_register(hit_points, ...)
    starpu_insert_task(codelets::advance_eye_paths,  eye_paths, seeds, hit_points);
    starpu_data_unregister_submit(eye_paths);

    ... // all other tasks for this iteration

    starpu_data_unregister_submit(hit_points);

    iteration++;
  }

  starpu_data_unregister(seeds);
}
