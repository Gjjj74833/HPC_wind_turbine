#!/bin/bash

### Driver for the Monte-Carlo simulation
### Developed by Yihan Liu, 02/08/2025

### This script takes one int argument, which run the simulation $n_simulation times. 
### If no argument is provided, 100 is the default.

n_simulations=20
n_arrays=50

### If a positive integer provided as argument, use this number
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        n_simulations=$(( $1/$n_arrays ))
    else
        echo "Argument is the number of simulation to run, must be a positive integer"
        exit 1
    fi
fi

module load python/3.9


for special_phase in {0..30}; do
    echo "Running for special_phase=${special_phase}"

    # Create necessary directories
    mkdir -p seeds_ablation_surge_${special_phase}_pi
    mkdir -p results_ablation_surge_${special_phase}_pi

    # Generate seeds
    python3 gen_seeds.py seeds_ablation_surge_${special_phase}_pi $1 

    ### ./run.sh $1={simulation number}, $2={simulation time}, $3={directory name}, $4={epsilon}, $5={iteration number}, $6={special phase index}
    sbatch monteCarlo.slurm $n_simulations $2 ablation_surge_${special_phase}_pi 0 0 $special_phase

done

