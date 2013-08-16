#!/usr/bin/env ruby

require './lib/common'

args = {scene: ['kitchen', 'cornell', 'luxball'],
        max_threads: [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64, 128, 256],
        photons_per_iter: [20,21,22]
}

KEPLER_ENV = 'CUDA_VISIBLE_DEVICES="0"'
FERMI_ENV  = 'CUDA_VISIBLE_DEVICES="1"'
BOTH_ENV   = 'CUDA_VISIBLE_DEVICES="0,1"'

run_all('rr-cpu', 'scalability', args)

run_all('rr-cuda',   'scalability-fermi',  args, FERMI_ENV)
run_all('rr-cuda',   'scalability-kepler', args, KEPLER_ENV)
run_all('rr-cuda',   'scalability-both',   args, BOTH_ENV)

run_all('rr-hybrid', 'scalability-fermi',  args, FERMI_ENV)
run_all('rr-hybrid', 'scalability-kepler', args, KEPLER_ENV)
run_all('rr-hybrid', 'scalability-both',   args, BOTH_ENV)
