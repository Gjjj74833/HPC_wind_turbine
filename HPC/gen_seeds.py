# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:43:03 2023
generating seeds 

@author: Yihan Liu
"""

import numpy as np

def generate_and_save_unique_arrays(n, n_array, seed=None):
    # Ensure n is within valid range and can be evenly divided by n_array
    if n % n_array != 0:
        raise ValueError("Invalid input: n is not divisible by n_array.")

    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate n*3 unique random integers within the desired range
    all_values = np.random.choice(np.arange(-9600000, 9600000), 
                                  size=n*3, replace=False)
    
    # Determine the number of rows per sub-array
    rows_per_array = n // n_array
    
    for i in range(n_array):
        # Select and reshape the values for each sub-array
        start_idx = i * rows_per_array * 3
        end_idx = start_idx + rows_per_array * 3
        sub_array = all_values[start_idx:end_idx].reshape((rows_per_array, 3))
        
        # Save the sub-array to a .npy file
        filename = f'./seeds/seeds_{i+1}.npy'
        np.save(filename, sub_array)
        print(f"Saved: {filename}")

# Example usage
generate_and_save_unique_arrays(10000, 50, seed=123)


seeds = loseeds = np.load(f'./seeds/seeds_4.npy')



