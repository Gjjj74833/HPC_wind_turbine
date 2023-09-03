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
    wind_speeds = [data['wind_speed'] for data in datas]
    wave_etas = [data['wave_eta'] for data in datas]
    Q_ts = [data['Q_t'] for data in datas]
    
    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    wind_speed = np.hstack(wind_speeds)
    wave_eta = np.hstack(wave_etas)
    Q_t = np.hstack(Q_ts)
    
    print("t", t.shape)
    print("wind_speed", wind_speed.shape)
    print("wave_eta", wave_eta.shape)
    print("state", state.shape)
    print("Qt", Q_t.shape)

        
    num_states = state.shape[1]

    for i in range(num_states):
        state_i = state[:, i, :]
        
        max_index = np.argmax(state_i, axis=1)
        min_index = np.argmin(state_i, axis=1)

        # Flatten the indices and count occurrences
        max_counts = np.bincount(max_index, minlength=state_i.shape[1])
        min_counts = np.bincount(min_index, minlength=state_i.shape[1])

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # Plot for max occurrences
        ax[0].bar(range(state_i.shape[1]), max_counts, color='b', alpha=0.7, label="Max occurrences")
        ax[0].set_title(f"Number of occurrences for Max in State {i}")
        ax[0].set_xlabel("Simulation Index")
        ax[0].set_ylabel("Occurrences")
        ax[0].legend()

        # Plot for min occurrences
        ax[1].bar(range(state_i.shape[1]), min_counts, color='r', alpha=0.7, label="Min occurrences")
        ax[1].set_title(f"Number of occurrences for Min in State {i}")
        ax[1].set_xlabel("Simulation Index")
        ax[1].set_ylabel("Occurrences")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(f"./results_figure/max_min_occurrences_histogram_state_{i}.png", dpi=600)
        plt.close(fig)  # Close the figure to free up memory
    
plot_quantiles()


    

