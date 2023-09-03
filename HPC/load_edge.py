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
    
    # record the index of simulation that have the most occurrence of max and min
    # value for each 7 states
    max_occ_sim = []
    min_occ_sim = []
    
    # record the index of simulation that have the max and min value through all
    # time steps for each 7 states
    max_value_sim = []
    min_value_sim = []
    
    for i in range(7):
        state_i = state[:, i, :]
        
        # store the max and min value for ith state in all time steps 
        max_value = np.max(state_i, axis=1)
        min_value = np.min(state_i, axis=1)
        
        # find the time step where the max and min occur
        max_value_time = np.argmax(max_value)
        min_value_time = np.argmin(min_value)
 
        # store the simulation index that have the most occurrence of max
        # and min at each time step
        max_index = np.argmax(state_i, axis=1)
        min_index = np.argmin(state_i, axis=1)
        
        # find the simulation index for the max and min value occured over
        # the entire time domain
        max_value_sim.append(max_index[max_value_time])
        min_value_sim.append(min_index[min_value_time])
        
        print(f'shape max index {i}', max_index.shape)
        
        # from max_index, all time steps, count the occurrence of max and min
        # for each simulation
        max_counts = np.bincount(max_index, minlength=state_i.shape[1])
        min_counts = np.bincount(min_index, minlength=state_i.shape[1])
        
        print(f'shape max counts {i}', max_counts.shape)
        
        # for each state, find the simulation that have the most occurrence
        # of max and min value
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
    
    print("On the entire time domain, the max occured at index of simulation", max_value_sim )
    print("On the entire time domain, the max occured at index of simulation", max_value_sim )
    
    
plot_quantiles()


    

