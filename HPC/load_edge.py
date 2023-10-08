# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:16:33 2023
Load and visualize the simulation results

@author: Yihan Liu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle


def load_data():
    
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
    
    return t, state, wind_speed, wave_eta, Q_t

def plot_quantiles(t, state, wind_speed, wave_eta, Q_t):
    
    # Get the central 75% ####################
    # States
    percentile_87_5 = np.percentile(state, 87.5, axis=2)
    percentile_12_5 = np.percentile(state, 12.5, axis=2)

    # Wind speed
    wind_percentile_87_5 = np.percentile(wind_speed, 87.5, axis=1)
    wind_percentile_12_5 = np.percentile(wind_speed, 12.5, axis=1)
    
    # Wave elevation
    wave_percentile_87_5 = np.percentile(wave_eta, 87.5, axis=1)
    wave_percentile_12_5 = np.percentile(wave_eta, 12.5, axis=1)
    
    # Tension force
    Qt_percentile_87_5 = np.percentile(Q_t, 87.5, axis=1)
    Qt_percentile_12_5 = np.percentile(Q_t, 12.5, axis=1)
    
    # Get the central 25% ####################
    # States
    percentile_62_5 = np.percentile(state, 62.5, axis=2)
    percentile_37_5 = np.percentile(state, 37.5, axis=2)
    
    # Wind speed
    wind_percentile_62_5 = np.percentile(wind_speed, 62.5, axis=1)
    wind_percentile_37_5 = np.percentile(wind_speed, 37.5, axis=1)
    
    # Wave elevation
    wave_percentile_62_5 = np.percentile(wave_eta, 62.5, axis=1)
    wave_percentile_37_5 = np.percentile(wave_eta, 37.5, axis=1)
    
    # Tension force
    Qt_percentile_62_5 = np.percentile(Q_t, 62.5, axis=1)
    Qt_percentile_37_5 = np.percentile(Q_t, 37.5, axis=1)
    
    # Get the median (50%) ####################
    # States
    percentile_50 = np.percentile(state, 50, axis=2)
    
    # Wind speed
    wind_percentile_50 = np.percentile(wind_speed, 50, axis=1)
    
    # Wave elevation
    wave_percentile_50 = np.percentile(wave_eta, 50, axis=1)
    
    # Tension force
    Qt_percentile_50 = np.percentile(Q_t, 50, axis=1)

    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor Speed (rpm)']
    
    start_time = 0
    end_time = t[-1]
    
    # Create one big figure
    fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(13, 22))
    
    # Flatten the axes array
    axes = axes.flatten()
    
    # First subplot for wind speed
    axes[0].fill_between(t, wind_percentile_12_5, wind_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    axes[0].fill_between(t, wind_percentile_37_5, wind_percentile_62_5, color='b', alpha=1)
    axes[0].plot(t, wind_percentile_50, color='r', linewidth=1)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Wind Speed (m/s)')
    axes[0].set_title('Time evolution of Wind Speed')
    axes[0].set_xlim(start_time, end_time)
    axes[0].grid(True)
    
    # Second subplot for wave_eta
    axes[1].fill_between(t, wave_percentile_12_5, wave_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    axes[1].fill_between(t, wave_percentile_37_5, wave_percentile_62_5, color='b', alpha=1)
    axes[1].plot(t, wave_percentile_50, color='r', linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Water Surface Elevation at x = 0 (m)')
    axes[1].set_title('Time evolution of Wave Surface Elevation at x = 0')
    axes[1].set_xlim(start_time, end_time)
    axes[1].grid(True)
    
    # subplots for states
    for i in range(7):
        ax = axes[2*i+2]
        ax.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        ax.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        ax.plot(t, percentile_50[:, i], color='r', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{state_names[i]}')
        ax.set_title(f'Time evolution of {state_names[i]}')
        ax.set_xlim(start_time, end_time)
        ax.grid(True)
        
        ax_short = axes[2*i+3]
        ax_short.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        ax_short.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        ax_short.plot(t, percentile_50[:, i], color='r', linewidth=1)
        ax_short.set_xlabel('Time (s)')
        ax_short.set_ylabel(f'{state_names[i]}')
        ax_short.set_title(f'Time evolution of {state_names[i]}')
        ax_short.set_xlim(end_time - 30, end_time)
        ax_short.grid(True)
        
    axes[16].fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    axes[16].fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    axes[16].plot(t, Qt_percentile_50, color='r', linewidth=1)
    axes[16].set_xlabel('Time (s)')
    axes[16].set_ylabel('Average Tension Force Per Line')
    axes[16].set_title('Time evolution of Average Tension Force Per Line (N)')
    axes[16].set_xlim(start_time, end_time)
    axes[16].grid(True)
    
    axes[17].fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    axes[17].fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    axes[17].plot(t, Qt_percentile_50, color='r', linewidth=1)
    axes[17].set_xlabel('Time (s)')
    axes[17].set_ylabel('Average Tension Force Per Line')
    axes[17].set_title('Time evolution of Average Tension Force Per Line (N)')
    axes[17].set_xlim(end_time - 30, end_time)
    axes[17].grid(True)
    
    n_simulation = wind_speed.shape[1]
    
    plt.tight_layout()
    plt.savefig(f'./results_figure/percentile_{n_simulation}simulations_{end_time}seconds.png')
    plt.close(fig)
    
    return percentile_87_5, percentile_12_5, percentile_62_5, percentile_37_5
    
    

def plot_trajectories(t, state, wind_speed, wave_eta):
    
    
    
    ######################################################################
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor Speed (rpm)']

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
    
    percentile_87_5 = np.percentile(state, 87.5, axis=2)
    percentile_12_5 = np.percentile(state, 12.5, axis=2)
    percentile_62_5 = np.percentile(state, 62.5, axis=2)
    percentile_37_5 = np.percentile(state, 37.5, axis=2)
    percentile_50 = np.percentile(state, 50, axis=2)
    max_state = np.max(state, axis=2)
    min_state = np.min(state, axis=2)
    
    for i in range(7):
        state_i = state[:, i, :]
        
        # find the time step where the max and min occur
        max_value_time = np.argmax(max_state[:, i])
        min_value_time = np.argmin(min_state[:, i])
 
        # store the simulation index that have the most occurrence of max
        # and min at each time step
        max_index = np.argmax(state_i, axis=1)
        min_index = np.argmin(state_i, axis=1)
        
        # find the simulation index for the max and min value occured over
        # the entire time domain
        max_value_sim.append(max_index[max_value_time])
        min_value_sim.append(min_index[min_value_time])
        
        # from max_index, all time steps, count the occurrence of max and min
        # for each simulation
        max_counts = np.bincount(max_index, minlength=state_i.shape[1])
        min_counts = np.bincount(min_index, minlength=state_i.shape[1])
        
        # for each state, find the simulation that have the most occurrence
        # of max and min value
        max_occ_sim.append(np.argmax(max_counts))
        min_occ_sim.append(np.argmax(min_counts))
        
        max_counts = max_counts*t[1]
        min_counts = min_counts*t[1]

        # Plot for max occurrences
        bin_edges = np.arange(0, max(max_counts)+6*t[1], 5*t[1])  
        ax[2*i].hist(max_counts, bins=bin_edges, density=True, alpha=0.7, align='mid')
        ax[2*i].set_title(f'State {i} - Max Occurrences')
        ax[2*i].set_xlabel('Time Stay at Max (s)')
        ax[2*i].set_ylabel('Frequency')
        selected_ticks = bin_edges[::8]  # select every second bin edge for x-ticks
        ax[2*i].set_xticks(selected_ticks)
        ax[2*i].tick_params(axis='x', rotation=45)  # rotate x-ticks for better readability

        # Plot for min occurrences
        bin_edges = np.arange(0, max(min_counts)+6*t[1], 5*t[1])  # Define bin edges for the histogram
        ax[2*i + 1].hist(min_counts, bins=bin_edges, density=True, alpha=0.7, align='mid')
        ax[2*i + 1].set_title(f'State {i} - Min Occurrences')
        ax[2*i + 1].set_xlabel('Time Stay at Min (s)')
        ax[2*i + 1].set_ylabel('Frequency')
        selected_ticks = bin_edges[::8]  # select every second bin edge for x-ticks
        ax[2*i + 1].set_xticks(selected_ticks)
        ax[2*i + 1].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig("./results_figure/max_min_occurrences_histogram_all_states.png", dpi=600)
    plt.close(fig)  
    
    print("The simulation that has the most occurrence of max is", max_occ_sim)
    print("The simulation that has the most occurrence of min is", min_occ_sim)
    
    print("On the entire time domain, the max occured at index of simulation", max_value_sim )
    print("On the entire time domain, the min occured at index of simulation", min_value_sim )
    
    # plot trajectories 
    
    def plot_helper(ax, index):
        
        # plot wind
        ax[0].plot(t, wind_speed[:, index], color='black', linewidth=0.5)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Time evolution of Wind Speed (m/s)')
        ax[0].set_ylabel('Wind speed (m/s)')
        ax[0].grid(True)
        ax[0].set_xlim(0, t[-1])
        
        # plot wave
        ax[1].plot(t, wave_eta[:, index], color='black', linewidth=0.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_title('Time evolution of Wave Surface Elevation at x = 0 (m)')
        ax[1].set_ylabel('Wave height (m)')
        ax[1].grid(True)
        ax[1].set_xlim(0, t[-1])
        
        # plot 7 states
        for j in range(7):
            ax[j+2].plot(t, max_state[:,j], alpha=0.6, linewidth=0.5)
            ax[j+2].plot(t, min_state[:,j], alpha=0.6, linewidth=0.5)

            ax[j+2].plot(t, state[:, j, index], color='black', linewidth=0.5)
            ax[j+2].set_xlabel('Time (s)')
            ax[j+2].set_ylabel(f'{state_names[j]}')
            
            ax[j+2].fill_between(t, percentile_12_5[:, j], percentile_87_5[:, j], color='b', alpha=0.3, edgecolor='none')
            ax[j+2].fill_between(t, percentile_37_5[:, j], percentile_62_5[:, j], color='b', alpha=0.3)
            ax[j+2].plot(t, percentile_50[:, j], color='r', alpha=0.8, linewidth=0.5)
            
            ax[j+2].set_title(f'Time evolution of {state_names[j]}')
            ax[j+2].grid(True)
            ax[j+2].set_xlim(0, t[-1])
            
        ax[9].remove()
      
    
    # for 7 states:
    for i in range(7):
        # create subplots for each simulation index in max_occ_sim
        fig_max_occ, ax_max_occ = plt.subplots(5, 2, figsize=(15, 20))
        fig_max_occ.suptitle(f'Trajectories for the simulation that have the most occurrence for {state_names[i]} max value from {state.shape[2]} simulations', fontsize=16)
        ax_max_occ = ax_max_occ.flatten()
        
        plot_helper(ax_max_occ, max_occ_sim[i])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(f'./results_figure/max_occ_{state_names[i]}.png', dpi=600)
        plt.close(fig_max_occ) 
        
        
        # create subplots for each simulation index in mix_occ_sim
        fig_min_occ, ax_min_occ = plt.subplots(5, 2, figsize=(15, 20))
        fig_min_occ.suptitle(f'Trajectories for the simulation that have the most occurrence for {state_names[i]} min value from {state.shape[2]} simulations', fontsize=16)
        ax_min_occ = ax_min_occ.flatten()
        
        plot_helper(ax_min_occ, min_occ_sim[i])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(f'./results_figure/min_occ_{state_names[i]}.png', dpi=600)
        plt.close(fig_min_occ) 
        
        # create subplots for each simulation index in max_value_sim
        fig_max_value, ax_max_value = plt.subplots(5, 2, figsize=(15, 20))
        fig_max_value.suptitle(f'Trajectories for the simulation that have the maximum value for {state_names[i]} from {state.shape[2]} simulations', fontsize=16)
        ax_max_value = ax_max_value.flatten()
        
        plot_helper(ax_max_value, max_value_sim[i])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(f'./results_figure/max_value_{state_names[i]}.png', dpi=600)
        plt.close(fig_max_value) 
        
        # create subplots for each simulation index in min_value_sim
        fig_min_value, ax_min_value = plt.subplots(5, 2, figsize=(15, 20))
        fig_min_value.suptitle(f'Trajectories for the simulation that have the minimum value for {state_names[i]} from {state.shape[2]} simulations', fontsize=16)
        ax_min_value = ax_min_value.flatten()
        
        plot_helper(ax_min_value, min_value_sim[i])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(f'./results_figure/min_value_{state_names[i]}.png', dpi=600)
        plt.close(fig_min_value) 
    
    
def pitchAnaly(state):
    pitch = state[::20, 4, :]
    pitch_rate = state[::20, 5, :]

    # Reshape data
    all_pitch = pitch.reshape(-1)
    all_rate = pitch_rate.reshape(-1)

    # Kernel Density Estimation for pitch and pitch_rate
    kde_pitch = gaussian_kde(all_pitch)
    kde_rate = gaussian_kde(all_rate)

    # Create linspace for plotting
    x_pitch = np.linspace(min(all_pitch), max(all_pitch), 1000)
    x_rate = np.linspace(min(all_rate), max(all_rate), 1000)

    # Plot density
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))

    # Pitch Plot
    ax[0].hist(all_pitch, bins=100, density=True, alpha=0.6, color='r', label='Data')
    ax[0].plot(x_pitch, kde_pitch(x_pitch), 'k', lw=1, label='PDF')
    ax[0].set_title('Pitch Distribution')
    ax[0].set_xlabel('Pitch (deg)')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Pitch Rate Plot
    ax[1].hist(all_rate, bins=100, density=True, alpha=0.6, color='b', label='Data')
    ax[1].plot(x_rate, kde_rate(x_rate), 'k', lw=1, label='PDF')
    ax[1].set_title('Pitch Rate Distribution')
    ax[1].set_xlabel('Pitch Rate (deg/s)')
    ax[1].set_ylabel('Density')
    ax[1].legend()

    plt.savefig('./results_figure/pitch_distribution.png', dpi=600)
    plt.tight_layout()
    plt.close()

    # Save PDFs using pickle
    with open('./density_function/kde_pitch.pkl', 'wb') as f:
        pickle.dump(kde_pitch, f)

    with open('./density_function/kde_rate.pkl', 'wb') as f:
        pickle.dump(kde_rate, f)
    
def extremeOccurDen(state):
    
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor Speed (rpm)']
    
    max_state = np.max(state, axis=2)
    min_state = np.min(state, axis=2)
    
    fig, ax = plt.subplots(7, 2, figsize=(15, 30))
    ax = ax.flatten()
    fig.suptitle('Distribution of Extreme Values for Each State from Monte Carlo Simulation', fontsize=16)
    for i in range(7):
        
        kde_max = gaussian_kde(max_state[:,i])
        kde_min = gaussian_kde(min_state[:,i])
        
        with open('./density_function/max_kde_{i}.pkl', 'wb') as f:
            pickle.dump(kde_max, f)

        with open('./density_function/min_kde_{i}.pkl', 'wb') as f:
            pickle.dump(kde_min, f)
        
        x_max = np.linspace(min(max_state[:,i]), max(max_state[:,i]), 1000)
        x_min = np.linspace(min(min_state[:,i]), max(min_state[:,i]), 1000)
        
        ax[2*i].hist(max_state[:,i], bins=100, density=True, alpha=0.6, color='r', label='Max Data')
        ax[2*i].plot(x_max, kde_max(x_max), 'k', lw=1, label='PDF')
        ax[2*i].set_xlabel(state_names[i])
        ax[2*i].set_ylabel('Density')
        ax[2*i].legend()
        ax[2*i].grid(True, linestyle='--', alpha=0.7)

        # Pitch Rate Plot
        ax[2*i+1].hist(min_state[:,i], bins=100, density=True, alpha=0.6, color='b', label='Min Data')
        ax[2*i+1].plot(x_min, kde_min(x_min), 'k', lw=1, label='PDF')
        ax[2*i+1].set_xlabel(state_names[i])
        ax[2*i+1].set_ylabel('Density')
        ax[2*i+1].legend()
        ax[2*i].grid(True, linestyle='--', alpha=0.7)
        
    plt.savefig('./results_figure/extreme_distribution.png', dpi=600)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.close()
        
    
    

t, state, wind_speed, wave_eta, Q_t = load_data()

pitchAnaly(state)
extremeOccurDen(state)


    

