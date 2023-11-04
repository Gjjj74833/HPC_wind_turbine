# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:08:56 2023

@author: ghhh7
"""
import numpy as np
import subprocess

def genWind(v_w, end_time, time_step, seed):
    """
    Use Turbsim to generate a wind with turbulence.

    Parameters
    ----------
    v_w : float
        the average wind speed
    end_time : float
        the time to analysis. Should be consistent with the model driver
    time_step : float
        the time step to analysis. Should be consistent with the model driver

    Returns
    -------
    horSpd : list
        A list of horizontal wind speed computed at each time step

    """
    end_time += 1
        
    # Generate seeds for random wind model
    seed1 = np.random.randint(-2147483648, 2147483648)
    seed2 = np.random.randint(-2147483648, 2147483648)
    seed = [seed1, seed2]
    path_inp = +'TurbSim.inp'
    
    
    # Open the inp file and overwrite with given parameters
    with open(path_inp, 'r') as file:
        lines = file.readlines()
        
    # Overwrite with new seeds
    line = lines[4].split()
    line[0] = str(seed[0])
    lines[4] = ' '.join(line) + '\n'

    line = lines[5].split()
    line[0] = str(seed[1])
    lines[5] = ' '.join(line) + '\n'
    
    # Overwrite "AnalysisTime" and "UsableTime"
    line = lines[21].split()
    line[0] = str(end_time)
    lines[21] = ' '.join(line) + '\n'
    
    # Overwrite the "TimeStep "
    line = lines[20].split()
    line[0] = str(time_step)
    lines[20] = ' '.join(line) + '\n'
    
    # Overwrite the average reference wind velocity
    line = lines[39].split()
    line[0] = str(v_w)
    lines[39] = ' '.join(line) + '\n'
    
    # Update the input file
    with open(path_inp, 'w') as file:
        file.writelines(lines)
    
    # Run the Turbsim to generate wind
    command = ["turbsim", path_inp]
    subprocess.run(command)
    
    # Read the output file
    path_hh = f'./turbsim/TurbSim_{sys.argv[1]}/TurbSim_{file_index}.hh'
    
    with open(path_hh, 'r') as file:
        lines = file.readlines()
    
    # Skip the header
    data = lines[8:]
    
    horSpd = []

    for line in data:
        columns = line.split()
        horSpd.append(float(columns[1]))  
    

    return np.array(horSpd)