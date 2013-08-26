#!/usr/bin/env ruby

require './lib/common'

args = {scene: ['kitchen', 'cornell', 'luxball'],
        max_iters_at_once: [1,2,3,4],
        sched: ['peager', 'pheft']
}

NONE_ENV   = 'CUDA_VISIBLE_DEVICES=""'
KEPLER_ENV = 'CUDA_VISIBLE_DEVICES="0"'
FERMI_ENV  = 'CUDA_VISIBLE_DEVICES="1"'
BOTH_ENV   = 'CUDA_VISIBLE_DEVICES="0,1"'

run_all('sppmpa-starpu', 'sppmpa-starpu', 'iters_at_once-none',   args, NONE_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'iters_at_once-fermi',  args, FERMI_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'iters_at_once-kepler', args, KEPLER_ENV)
run_all('sppmpa-starpu', 'sppmpa-starpu', 'iters_at_once-both',   args, BOTH_ENV)
