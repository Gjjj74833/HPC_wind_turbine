# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 04:24:41 2025

Generate seeds for ablation test

@author: Yihan
"""


import numpy as np
import sys


    
    
def generate_and_save_unique_arrays(n, n_array, seed=None):
    # Generate unique values for the entire range
    unique_values = np.random.choice(np.arange(0, 9600000), 
                                     size=n*3, replace=False)

    rows_per_array = n // n_array
    
    first_component_list = np.linspace(-np.pi, np.pi, n_array)
    
    for i in range(n_array):
        
        
        # Fixed values for the second and third components
        first_component = 4021713
        second_component = 438654
        third_component = first_component_list[i]
        
        seed_array = np.column_stack((first_component, second_component, third_component))
        
        filename = f'./{sys.argv[1]}/seeds_{i+1}.npy'
        #filename = f'seeds_{i+1}.npy'
        np.save(filename, seed_array)
        print(f"Saved: {filename}")

generate_and_save_unique_arrays(int(sys.argv[2]), 50)

 
