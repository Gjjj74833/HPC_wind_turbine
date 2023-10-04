#!/bin/bash

### Driver for the Monte-Carlo simulation
### Developed by Yihan Liu, 08/29/2023

### This script takes one int argument, which run the simulation $n_simulation times. 
### If no argument is provided, 100 is the default.

n_simulations=20

### If a positive integer provided as argument, use this number
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        n_simulations=$(( $1 / 50 ))
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
    mkdir turbsim
    echo "Creating turbsim directory"
fi

### Back up old output if exists
if [ "$(ls -A results)" ]; then
    if [ ! -d "old_results" ]; then
        mkdir old_results
        echo "Creating directory to back up old results"
    fi

    timestamp=$(date +"%Y%m%d_%H%M%S")
    tar -czvf old_results/results_${timestamp}.tar.gz -C results .
    echo "Back up the previous outputs..."
    rm results/*
fi


# Submit the job and get the job ID
sbatch monteCarlo.slurm $n_simulations $2
