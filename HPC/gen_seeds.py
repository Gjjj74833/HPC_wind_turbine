# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:43:03 2023
generating seeds 

@author: Yihan Liu
"""

import numpy as np
import sys
    
def generate_and_save_unique_arrays(n, n_array, seed=None):
    # Generate unique values for the entire range
    unique_values = np.random.choice(np.arange(0, 9600000), 
                                     size=n*3, replace=False)

    rows_per_array = n // n_array

    for i in range(n_array):
        start_idx = i * rows_per_array * 3
        end_idx = start_idx + rows_per_array * 3
        
        # Get the sub-array for the current set
        sub_array = unique_values[start_idx:end_idx].reshape((rows_per_array, 3))
        
        filename = f'./{sys.argv[1]}/seeds_{i+1}.npy'
        np.save(filename, sub_array)
        print(f"Saved: {filename}")

generate_and_save_unique_arrays(int(sys.argv[3]), 50)



