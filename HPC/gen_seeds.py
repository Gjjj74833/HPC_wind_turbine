# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:43:03 2023
generating seeds 

@author: Yihan Liu
"""

import numpy as np

def generate_and_save_unique_arrays(n, n_array, seed=None):
    if n % n_array != 0:
        raise ValueError("Invalid input: n is not divisible by n_array.")

    np.random.seed(seed)
    
    # Generate the first column with the full range
    first_column = np.random.choice(np.arange(0, 9600000), 
                                    size=n, replace=False)

    # Generate the second column with only non-negative values
    second_column = np.random.choice(np.arange(0, 9600000), 
                                     size=n, replace=False)
    
    rows_per_array = n // n_array
    
    for i in range(n_array):
        start_idx_1col = i * rows_per_array
        end_idx_1col = start_idx_1col + rows_per_array
        
        # Get the sub-array for the first column
        sub_array_1col = first_column[start_idx_1col:end_idx_1col].reshape((rows_per_array, 1))
        
        # Get the sub-array for the second column
        sub_array_2col = second_column[start_idx_1col:end_idx_1col].reshape((rows_per_array, 1))
        
        # Combine them to form a sub-array with two unique elements per row
        sub_array = np.hstack((sub_array_1col, sub_array_2col))
        
        filename = f'./seeds/seeds_{i+1}.npy'
        np.save(filename, sub_array)
        print(f"Saved: {filename}")

generate_and_save_unique_arrays(10000, 50)
seeds = np.load(f'./seeds/seeds_4.npy')



