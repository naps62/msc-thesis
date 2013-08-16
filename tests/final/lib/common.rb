#!/usr/bin/env ruby

$LOAD_PATH.unshift File.dirname(__FILE__)
require 'fileutils'
require 'pry'

HOME      = ENV['HOME']
NODE      = `hostname`.split.first
ROOT      = "#{HOME}/projects/msc-thesis"
BIN_ROOT  = "#{HOME}/projects/msc-thesis/bin"
TEST_ROOT = "#{HOME}/projects/msc-thesis/tests/final"
KBEST     = "#{HOME}/projects/msc-thesis/tests/final/lib/kbest.rb"

def run_all(exec, test_name, args, env = "")
  combinations = args.values.first.product(*args.values[1..-1])

  combinations.each do |arg|
    args_name = arg.each_with_index.map { |value, i| "#{args.keys[i]}-#{value}"}.join('__')
    this_test_root = "#{TEST_ROOT}/#{exec}/#{test_name}/#{NODE}/#{args_name}"
    FileUtils.mkdir_p this_test_root

    kbest_ops   = [
      "--out #{this_test_root}",
      "--k 3",
      "--diff 0.05",
      "--min 3",
      "--max 10"
    ].join(' ')

    args_with_keys = arg.each_with_index.map { |value, i| "--#{args.keys[i]} #{value}"}.join('  ')
    cmd = "#{BIN_ROOT}/#{exec} #{args_with_keys} --output_dir #{this_test_root}"
    single_run(env, kbest_ops, cmd)
  end
end

def single_run(env, ops, cmd)
  full_cmd = "cd #{ROOT} && #{KBEST} #{ops} \"#{env} #{cmd}\""
  puts   full_cmd
  system full_cmd
end
