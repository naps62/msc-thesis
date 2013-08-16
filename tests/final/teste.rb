#!/usr/bin/env ruby

require './lib/common'

args = {scene: ['kitchen'],
        max_threads: [4],
        photons_iter: [15],
        max_iters: [2]
}

run_all('sppmpa-starpu', 'teste', args, "CUDA_VISIBLE_DEVICES=''")
