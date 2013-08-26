#!/usr/bin/env ruby

require './lib/common'

rr_args = {scene: ['kitchen', 'cornell', 'luxball'],
        max_threads: [1, 2, 4, 8, 16, 32, 64, 128, 256]
}
cpu_args    = rr_args.merge engine: [1]
gpu_args    = rr_args.merge engine: [2]
starpu_args = rr_args.merge engine: [0]

NONE_ENV   = 'CUDA_VISIBLE_DEVICES=""'
KEPLER_ENV = 'CUDA_VISIBLE_DEVICES="0"'
FERMI_ENV  = 'CUDA_VISIBLE_DEVICES="1"'
BOTH_ENV   = 'CUDA_VISIBLE_DEVICES="0,1"'

run_all('sppmpa-starpu', 'sppmpa-cpu', 'scalability', cpu_args, NONE_ENV)
run_all('sppmpa-starpu', 'sppmpa-cuda', 'scalability-fermi', gpu_args, FERMI_ENV)
run_all('sppmpa-starpu', 'sppmpa-cuda', 'scalability-kepler', gpu_args, KEPLER_ENV)

run_all('sppmpa-starpu', 'sppmpa-starpu', 'scalability-none',   starpu_args, NONE_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'scalability-fermi',  starpu_args, FERMI_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'scalability-kepler', starpu_args, KEPLER_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'scalability-both',   starpu_args, BOTH_ENV)
