#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mem=2gb

# Cluster Environment Setup
cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate RLSB3

python3 make_mesh.py

exit 0
