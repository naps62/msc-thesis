#!/usr/bin/env ruby

require './lib/common'

rr_args = {scene: ['kitchen', 'cornell', 'luxball'],
        max_threads: [1, 2, 4, 8, 16, 32, 64, 128, 256],
        photons_iter: [20],
        max_iters: [20],
}
cpu_args    = rr_args.merge engine: [1]
gpu_args    = rr_args.merge engine: [2]
starpu_args = rr_args.merge engine: [0]

NONE_ENV   = 'CUDA_VISIBLE_DEVICES=""'
KEPLER_ENV = 'CUDA_VISIBLE_DEVICES="0"'
FERMI_ENV  = 'CUDA_VISIBLE_DEVICES="1"'
BOTH_ENV   = 'CUDA_VISIBLE_DEVICES="0,1"'

#run_all('rr-cpu',        'rr-cpu',     'scalability', rr_args, NONE_ENV)
run_all('sppmpa-starpu', 'sppmpa-cpu', 'scalability', cpu_args, NONE_ENV)

#run_all('rr-cuda-single', 'rr-cuda1',     'scalability-fermi',  args, FERMI_ENV)
#run_all('rr-cuda-single', 'rr-cuda0',     'scalability-kepler', args, KEPLER_ENV)
#run_all('rr-cuda-both',   'rr-cuda-both', 'scalability-both',   args, BOTH_ENV)

#run_all('rr-hybrid-single', 'rr-hybrid1',     'scalability-fermi',  args, FERMI_ENV)
#run_all('rr-hybrid-single', 'rr-hybrid0',     'scalability-kepler', args, KEPLER_ENV)
#run_all('rr-hybrid-both',   'rr-hybrid-both', 'scalability-both',   args, BOTH_ENV)
