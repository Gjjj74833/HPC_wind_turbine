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

module load python/3.9
mkdir seeds_tension_n15_pi0_ite$3
mkdir results_tension_n15_pi0_ite$3
python3 gen_seeds.py seeds_tension_n15_pi0_ite$3 $1 

### ./run.sh $1={simulation number}, $2={simulation time}, $3={directory name}, $4={elispe pi/_}, $5={Iteration number} 
sbatch monteCarlo.slurm $n_simulations $2 tension_n15_pi0_ite$3 0 $3

