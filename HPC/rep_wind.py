# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:08:56 2023

@author: ghhh7
"""
import numpy as np
import subprocess

def genWind(seed, end_time, time_step=0.05, v_w=20):

    end_time += 1

    # Generate seeds for random wind model
    path_inp = 'TurbSim.inp'
    
    
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
    
    command = ["cp", "TurbSim.hh", f"./turbsim_output/{seed[0]}_{seed[1]}.hh"]
    subprocess.run(command)
    
seeds = [[-121491204, -1304678860],
         [-1424433141, 2045330612],
         [1365634968, 1998175349],
         [1959064098, -1690322664],
         [400672822, -1387694271],
         [631843952, 414653701],
         [384934251, 1838707713],
         [-1741971552, 560505679],
         [-514258701, 1239123680],
         [2062356962, -1973551217],
         [1351979082, -1358014331],
         [-118014334, -1217641822],
         [1005984867, -686930290],
         [-1726641590, 1688144086],
         [-464316375, 2044166783],
         [159058000, 1370406950],
         [163203302, -812798221],
         [416290233, -458643487],
         [1648937011, 1892887397],
         [-856074495, 432459576],
         [-2056281993, 1247348841],
         [-1793101032, 1508361815],
         [-967220521, 626952838],
         [-1772727346, 209617217],
         [1820087526, 32059213],
         [-1283202251, 1363511765],
         [-150537734.  -78472730],
         [1351979082. -1358014331],
         [1948384108. -656727226]]
        
for seed in seeds:
    genWind(seed, 3000, time_step=0.05, v_w=20)

###########################################################################################



