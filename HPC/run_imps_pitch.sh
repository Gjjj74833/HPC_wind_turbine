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



### Create results directory if the destination of the output directory does not exist.
if [ ! -d "results" ]; then
    mkdir results
    echo "Creating output directory for multi tasks..."
fi

if [ ! -d "turbsim" ]; then
-    mkdir turbsim
    echo "Creating turbsim directory"
fi

### Back up old output if exists
###if [ "$(ls -A results)" ]; then
###    if [ ! -d "old_results" ]; then
###        mkdir old_results
###        echo "Creating directory to back up old results"
###    fi

###    timestamp=$(date +"%Y%m%d_%H%M%S")
###    tar -czvf old_results/results_${timestamp}.tar.gz -C results .
###    echo "Back up the previous outputs..."
###    rm results/*
###fi

### create direcories and files for turbsim input
###if [ -d "turbsim" ]; then
###    echo "Deleting old turbsim workspaces/files..."
###    rm -rf turbsim
###fi

###mkdir turbsim
###echo "Creating work spaces for turbsim..."
###for i in $(seq 1 $n_arrays); do
###    mkdir turbsim/TurbSim_${i}
###    ### copy files
###    for file_index in $(seq 0 $(($n_simulations-1))); do
###        cp ./TurbSim.inp ./turbsim/TurbSim_${i}/TurbSim_${file_index}.inp
###    done
###done

### Submit the job and get the job ID
module load python/3.9
### ./run.sh $1 -> {simulation number}, $2={simulation time}, $3={samping seed ID}, $4={elispe pi/_}, $5={sampling seed}
### gen_seeds $3 $4
python3 gen_seeds.py 2 1 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 2 1 7703243

python3 gen_seeds.py 2 2 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 2 2 7703243

python3 gen_seeds.py 2 4 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 2 4 7703243

python3 gen_seeds.py 2 6 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 2 6 7703243

python3 gen_seeds.py 2 8 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 2 8 7703243
#####################################################
python3 gen_seeds.py 3 1 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 3 1 4033977

python3 gen_seeds.py 3 2 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 3 2 4033977

python3 gen_seeds.py 3 4 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 3 4 4033977

python3 gen_seeds.py 3 6 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 3 6 4033977

python3 gen_seeds.py 3 8 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 3 8 4033977
#####################################################
python3 gen_seeds.py 4 1 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 4 1 3071765

python3 gen_seeds.py 4 2 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 4 2 3071765

python3 gen_seeds.py 4 4 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 4 4 3071765

python3 gen_seeds.py 4 6 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 4 6 3071765

python3 gen_seeds.py 4 8 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 4 8 3071765
#####################################################
python3 gen_seeds.py 5 1 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 5 1 8242234

python3 gen_seeds.py 5 2 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 5 2 8242234

python3 gen_seeds.py 5 4 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 5 4 8242234

python3 gen_seeds.py 5 6 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 5 6 8242234

python3 gen_seeds.py 5 8 $n_simulations
sbatch monteCarlo.slurm $n_simulations $2 5 8 8242234