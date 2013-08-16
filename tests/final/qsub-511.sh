#!/bin/sh
#PBS -V
#PBS -N naps62-511
#PBS -l nodes=1:r511:ppn=24
#PBS -l walltime=20:00:00
#PBS -m bea
#PBS -e naps62-511.err
#PBS -o naps62-511.out

./qsub-common "$@"
