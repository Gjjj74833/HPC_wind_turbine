#!/bin/bash

### Create results directory if the destination of the output directory does not exist.
if [ ! -d "results" ]; then
    mkdir results
    echo "Creating output directory for multi tasks..."
fi

### Back up old output if exists
if [ "$(ls -A results)" ]; then
    if [ ! -d "old_results" ]; then
        mkdir old_results
        echo "Creating directory to back up old results"
    fi

    timestamp=$(date +"%Y%m%d_%H%M%S")
    tar -cvf old_results/results_${timestamp}.tar -C results .
    echo "Back up the previous outputs..."
    rm results/*
fi


sbatch monteCarlo.slurm

