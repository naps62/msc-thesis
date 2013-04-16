#!/usr/bin/env ruby

require 'fileutils'

HOME      = ENV['HOME']
NODE      = `hostname`.split.first
KBEST     = "#{HOME}/projects/beast/blotter/kbest.rb"
TEST_ROOT = "#{HOME}/projects/msc-thesis/tests/ppm-rr/scalability2"

THREADS = (1..16).each { |t| t * 2 }
SCENES = ['kitchen', 'cornell', 'alloy', 'bigmonkey', 'simple-mat', 'luxball']

THREADS.each do |t|
  SCENES.each do |scene|
    puts "  THREADS = #{t}, SCENE = #{scene}"
    this_test_root = "#{TEST_ROOT}/#{NODE}/#{scene}_t#{'%03d' % t}"
    FileUtils.mkdir_p this_test_root

    kbest_ops   = [
      "--out #{this_test_root}",
      "--k 3",
      "--diff 0.05",
      "--min 10",
      "--max 30"
    ].join(' ')

    cmd = [
      "#{HOME}/projects/msc-thesis/bin/ppm-rr",
      "--config #{TEST_ROOT}/common.cfg",
      "--max_threads #{t}",
      "--output_dir #{this_test_root}",
      "--scene_dir scenes/#{scene}",
      "--scene_file #{scene}.scn"
    ].join(' ')

    puts   "#{KBEST} #{kbest_ops} \"#{cmd}\""
    system "#{KBEST} #{kbest_ops} \"#{cmd}\""
  end
end
