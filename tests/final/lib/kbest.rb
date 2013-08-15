#!/usr/bin/env ruby

require 'trollop'
require 'open3'
require 'set'
require 'fileutils'

$opts = Trollop::options do
  opt :k,       'number of results to consider',                  default: 3
  opt :diff,    '% of difference between the k selected results', default: 0.05
  opt :min,     'minimum number of executions',                   default: 1,        short: :m
  opt :max,     'maximum number of executions',                   default: 20,       short: :M
  opt :reverse, 'are we looking for the highest instead?',        default: false,    short: :r
  opt :out,     'output directory',                               default: 'kbest/', short: :o
end

command = ARGV.join ' '

class Execution
  attr_accessor :value, :out, :err, :thread

  def initialize(command)
    stdin, out, err, @thread = Open3.popen3(command)
    @out = out.readlines
    @err = err.readlines
    @value = @err.last.to_f
  end

  def <=>(other)
    ret = @value <=> other.value
    return @thread.pid <=> other.thread.pid if ret.zero?
    return  ret unless $opts[:reverse]
    return -ret
  end
end


def done?(execs, values, diff, params)
  return false if params[:min] > 0 && execs <  params[:min] # at least #{min} runs made
  return true  if params[:max] > 0 && execs >= params[:max] # at most  #{max} runs made
  return true  if values >= params[:k]
  return false
end


execs = SortedSet.new
k     = $opts[:k]

begin
  puts "    run #[#{execs.size}]"
  # call the program
  execs.add(Execution.new(command))
  # only the value of the first k results is relevant
  values = execs.to_a[0..k].map(&:value)
  # results are ordered, so the last diff is enough to check if we have k good results
  diff = (values.last - values.first) / values.first
end until done?(execs.size, values.size, diff, $opts)

final = execs.to_a[0..k]

out_root = $opts[:out]
FileUtils.mkdir_p out_root
times_output = File.open("#{out_root}/kbest_times.csv", 'w')
final.each_with_index do |exec, i|
  times_output << "#{i}, #{exec.value}\n"

  {out: exec.out, err: exec.err}.each_pair do |ext, stream|
    file = File.open "#{out_root}/#{i}.#{ext}", "w"
    file << stream.join
    file.close
  end
end

times_output.close
