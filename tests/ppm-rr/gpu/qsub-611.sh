#!/bin/sh
#PBS -V
#PBS -N ppm-rr-scal-611
#PBS -l nodes=1:r611:ppn=24
#PBS -l walltime=20:00:00
#PBS -m bea
#PBS -e ppm-rr-scal-611.err
#PBS -o ppm-rr-scal-611.out


source /etc/profile.d/env-modules.sh
module load gcc/4.6.3
module load boost/1.49.0
module load cuda-5.0
module load freeimage

cd $HOME/projects/msc-thesis
echo $PWD
echo $PATH
ls /share/edu-mei
tests/ppm-rr/gpu/job.rb 
