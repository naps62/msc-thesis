#!/bin/sh
#PBS -V
#PBS -N ppm-rr-scal-511
#PBS -l nodes=1:r511:ppn=24
#PBS -l walltime=20:00:00
#PBS -m bea
#PBS -e ppm-rr-scal-511.err
#PBS -o ppm-rr-scal-511.out


ls /share/edu-mei > /dev/null
source /etc/profile.d/env-modules.sh
module load gcc/4.6.3
module load boost/1.49.0
module load cuda-5.0
module load freeimage

module list
#echo $PATH 1>&2
#echo $LD_LIBRARY_PATH 1>&2

cd $HOME/projects/msc-thesis
tests/ppm-rr/gpu_after_fix/job.rb 
