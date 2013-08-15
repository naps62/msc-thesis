#!/usr/bin/env ruby

require '../common'

args = {scene: ['kitchen', 'cornell', 'luxball'],
        max_threads: [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64, 128, 256],
        photons_per_iter: [20,21,22]
}

run_all('rr-hybrid', args)
