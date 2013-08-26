#!/usr/bin/env ruby

require './lib/common'

args = {scene: ['kitchen', 'cornell', 'luxball', 'alloy'],
        photons_iter: [19,20,21,22],
        max_iters: [500],
        save_offset: [20],
        sched: ['pheft']
}

NONE_ENV   = 'CUDA_VISIBLE_DEVICES=""'
KEPLER_ENV = 'CUDA_VISIBLE_DEVICES="0"'
FERMI_ENV  = 'CUDA_VISIBLE_DEVICES="1"'
BOTH_ENV   = 'CUDA_VISIBLE_DEVICES="0,1"'

run_all('sppmpa-starpu', 'sppmpa-starpu', 'quality', args, BOTH_ENV)
