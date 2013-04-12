#!/bin/sh
#PBS -V
#PBS -N ppm-rr-scal-611
#PBS -l nodes=1:r611:ppn=24
#PBS -l walltime=5:00:00
#PBS -m bea
#PBS -e ppm-rr-scal-611.err
#PBS -o ppm-rr-scal-611.out

source /etc/profile.d/env-modules.sh
module load cuda-5.0
module load gcc/4.6
module load boost/49
module load freeimage

cd $HOME/projects/msc-thesis
tests/ppm-rr/scalability/job.rb
