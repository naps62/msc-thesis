#!/usr/bin/env ruby

require 'fileutils'

HOME      = ENV['HOME']
NODE      = `hostname`.split.first
KBEST     = "#{HOME}/projects/beast/blotter/kbest.rb"
TEST_ROOT = "#{HOME}/projects/msc-thesis/tests/ppm-rr/scalability"

THREADS = [1, 2, 4, 6, 8, 12, 16, 24, 32]
PPI     = [19, 20, 21, 22]

EXEC = "#{{}}"

THREADS.each do |t|
  PPI.each do |ppi|
    puts "  THREADS = #{t}, PPI = #{ppi}"
    this_test_root = "#{TEST_ROOT}/#{NODE}/t#{t}_ppi#{ppi}"
    FileUtils.mkdir_p this_test_root

    kbest_ops   = [
      "--out #{this_test_root}",
      "--k 3",
      "--diff 0.05",
      "--min 3",
      "--max 30"
    ].join(' ')

    cmd = [
      "#{HOME}/projects/msc-thesis/bin/ppm-rr",
      "--config #{TEST_ROOT}/common.cfg",
      "--max_threads #{t}",
      "--photons_iter #{ppi}",
      "--output_dir #{this_test_root}",
    ].join(' ')

    puts   "#{KBEST} #{kbest_ops} \"#{cmd}\""
    system "#{KBEST} #{kbest_ops} \"#{cmd}\""
  end
end
