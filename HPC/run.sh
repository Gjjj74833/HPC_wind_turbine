#!/bin/bash

### Driver for the Monte-Carlo simulation
### Developed by Yihan Liu, 08/29/2023

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
mkdir results_ropeMCMC
mkdir seeds_ropeMCMC
module load python/3.9 
python3 gen_seeds.py seeds_ropeMCMC $1 
sbatch monteCarlo.slurm $n_simulations $2 