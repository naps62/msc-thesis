#!/bin/sh
#PBS -V
#PBS -N naps62-611
#PBS -l nodes=1:r611:ppn=24
#PBS -l walltime=10:00:00
#PBS -m bea
#PBS -e naps62-611.err
#PBS -o naps62-611.out

./qsub-common "$@"
