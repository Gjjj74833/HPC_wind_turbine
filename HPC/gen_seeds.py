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
    
    # Generate the first two columns with the full range
    first_two_columns = np.random.choice(np.arange(-9600000, 9600000), 
                                         size=n*2, replace=False)

    # Generate the third column with only non-negative values
    third_column = np.random.choice(np.arange(0, 9600000), 
                                    size=n, replace=False)
    
    rows_per_array = n // n_array
    
    for i in range(n_array):
        start_idx_2col = i * rows_per_array * 2
        end_idx_2col = start_idx_2col + rows_per_array * 2
        
        # Get the sub-array for the first two columns
        sub_array_2col = first_two_columns[start_idx_2col:end_idx_2col].reshape((rows_per_array, 2))
        
        start_idx_3col = i * rows_per_array
        end_idx_3col = start_idx_3col + rows_per_array
        
        # Get the sub-array for the third column
        sub_array_3col = third_column[start_idx_3col:end_idx_3col].reshape((rows_per_array, 1))
        
        # Combine them to form a sub-array with the third column non-negative
        sub_array = np.hstack((sub_array_2col, sub_array_3col))
        
        filename = f'./seeds/seeds_{i+1}.npy'
        np.save(filename, sub_array)
        print(f"Saved: {filename}")

generate_and_save_unique_arrays(10000, 50)
seeds = loseeds = np.load(f'./seeds/seeds_4.npy')



