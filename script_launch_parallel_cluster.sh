#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=66:mem=50gb

# If run no CPU, remove 'ngpus' and 'gpu_type' from above 


cd $PBS_O_WORKDIR

# Cluster Environment Setup
module load anaconda3/personal
source activate py38
NUM_PORT=65

# Run the python code
python3 launch_parallel_training.py -n $NUM_PORT

echo "Launched training!"

exit 0
