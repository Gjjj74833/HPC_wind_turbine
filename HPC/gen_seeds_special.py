# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:43:03 2023
generating seeds 

@author: Yihan Liu
"""

import numpy as np
    
    
def generate_and_save_unique_arrays(n, n_array, seed_1, seed_2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate unique values for the third elements
    unique_third_elements = np.random.choice(np.arange(0, 9600000), 
                                             size=n, replace=False)
    
    rows_per_array = n // n_array

    for i in range(n_array):
        start_idx = i * rows_per_array
        end_idx = start_idx + rows_per_array
        
        # Get the sub-array for the current set
        third_elements = unique_third_elements[start_idx:end_idx]
        
        # Create array with fixed first two elements and random third element
        sub_array = np.column_stack((np.full(rows_per_array, seed_1), 
                                     np.full(rows_per_array, seed_2), 
                                     third_elements))
        
        filename = f'./seeds/seeds_{i+1}.npy'
        np.save(filename, sub_array)
        print(f"Saved: {filename}")
        
        
# Example usage:
generate_and_save_unique_arrays(10000, 50, seed_1=5386811, seed_2=9035970)
seeds = np.load(f'./seeds/seeds_32.npy')
#print(seeds)




