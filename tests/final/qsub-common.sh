#!/bin/sh

source /etc/profile.d/env-modules.sh
module load gcc/4.6
module load boost/49
module load cuda-5.0
module load freeimage
mdule load hwloc
module load starpu

cd $HOME/projects/msc-thesis
ls /share/edu-mei > /dev/null
echo "Running tests/final/$1"
tests/final/$1
