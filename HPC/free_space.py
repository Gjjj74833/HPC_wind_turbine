# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 00:56:33 2024

@author: Yihan Liu
"""

import os
import numpy as np

def remove_slices_from_npz(directory, slices_to_remove):
    """
    Remove specified slices from the 'state' array in all .npz files in the directory.
    
    Parameters:
    directory (str): The directory containing the .npz files.
    slices_to_remove (list of int): The indices of slices to remove from the 'state' array.
    """
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            # Load the .npz file
            data = np.load(filepath)
            
            # Extract the state array
            state = data['state']
            
            # Remove specified slices
            state = np.delete(state, slices_to_remove, axis=1)
            
            t = data['t']
            new_t = np.array([t[1], t[-1]]) #[time step, end time]
                            
            # Save the modified data back to the .npz file
            np.savez(filepath, 
                     t=new_t,  
                     state=state, 
                     wind_speed=data['wind_speed'], 
                     wave_eta=data['wave_eta'], 
                     betas=data['betas'], 
                     seeds=data['seeds'], 
                     T_E=data['T_E'],
                     P_A=data['P_A'])
            print(f"Updated {filename} in {directory}")

def process_all_results_directories(base_directory, slices_to_remove):
    """
    Process all directories starting with 'results' in the base directory.
    
    Parameters:
    base_directory (str): The base directory containing the 'results' directories.
    slices_to_remove (list of int): The indices of slices to remove from the 'state' array.
    """
    
    # Iterate through all directories in the base directory
    for directory in os.listdir(base_directory):
        if directory.startswith('results') and os.path.isdir(os.path.join(base_directory, directory)):
            print(f"Processing directory: {directory}")
            remove_slices_from_npz(os.path.join(base_directory, directory), slices_to_remove)



# Base directory containing the results directories
base_directory = './'  # Change this to your base directory path

# Indices of the slices to remove
slices_to_remove = [1, 3, 5]

# Run the function to update all relevant directories
remove_slices_from_npz('results_surge_5_pi1', slices_to_remove)
#process_all_results_directories(base_directory, slices_to_remove)




