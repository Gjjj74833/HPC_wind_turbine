# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:16:33 2023
Load the simulation results

@author: Yihan Liu
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_quantiles():
    
    directory = 'results'
    
    # collect all data files
    data_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
    
    # load data from the files
    datas = [np.load(file) for file in data_files]
    
    t = datas[0]['t']
    
    states = [data['state'] for data in datas]

    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    
    ######################################################################
    state_names = ['Surge', 'Surge_Velocity', 'Heave', 'Heave_Velocity', 
                   'Pitch_Angle', 'Pitch_Rate', 'Rotor_speed']

    # Create one large subplot for max and min occurrences for each state
    fig, ax = plt.subplots(7, 2, figsize=(15, 30))
    ax = ax.flatten()  # Flatten to easily index
    
    max_occ_sim = []
    min_occ_sim = []
    
    max_value_sim = []
    min_value_sim = []
    
    for i in range(7):
        state_i = state[:, i, :]
        
        max_value = np.argmax(state_i, axis=1)
        mmin_value = np.argmin(state_i, axis=1)
        
 
        
        max_index = np.argmax(state_i, axis=1)
        min_index = np.argmin(state_i, axis=1)
        
        print(f'shape max index {i}', max_index.shape)
        
        # Count occurrences
        max_counts = np.bincount(max_index, minlength=state_i.shape[1])
        min_counts = np.bincount(min_index, minlength=state_i.shape[1])
        
        print(f'shape max counts {i}', max_counts.shape)
        
        max_occ_sim.append(np.argmax(max_counts))
        min_occ_sim.append(np.argmax(min_counts))

        # Plot for max occurrences
        ax[2*i].bar(range(state_i.shape[1]), max_counts, color='b', alpha=0.7, label="Max occurrences")
        ax[2*i].set_title(f"Number of occurrences for Max in {state_names[i]}")
        ax[2*i].set_xlabel("Simulation Index")
        ax[2*i].set_ylabel("Occurrences")
        ax[2*i].legend()

        # Plot for min occurrences
        ax[2*i+1].bar(range(state_i.shape[1]), min_counts, color='r', alpha=0.7, label="Min occurrences")
        ax[2*i+1].set_title(f"Number of occurrences for Min in {state_names[i]}")
        ax[2*i+1].set_xlabel("Simulation Index")
        ax[2*i+1].set_ylabel("Occurrences")
        ax[2*i+1].legend()
        
    plt.tight_layout()
    plt.savefig("./results_figure/max_min_occurrences_histogram_all_states.png", dpi=600)
    plt.close(fig)  
    
    print("The simulation that has the most occurrence of max is", max_occ_sim)
    print("The simulation that has the most occurrence of max is", min_occ_sim)
    
plot_quantiles()


    

