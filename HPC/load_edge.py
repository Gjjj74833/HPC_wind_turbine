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
from matplotlib.lines import Line2D
import seaborn as sns
import binsreg
import pandas as pd



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
    betas = [data['betas'] for data in datas]
    seeds = [data['seeds'] for data in datas]
    
    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    wind_speed = np.hstack(wind_speeds)
    wave_eta = np.hstack(wave_etas)
    seeds = np.hstack(seeds)
    
    return t, state, wind_speed, wave_eta, seeds

def merge_pitch_acc(states):
    """
    merge pitch acceleration to states
    states.shape: (time_step, state, simulation_index)
    state in this order:
    [surge, surge_velocity, heave, heave_velocity, pitch, pitch_rate, pitch_acceleration, rotor_speed]
    
    return the merged state
    """
    pitch_rate = states[:, 5, :]  
    pitch_acceleration = np.diff(pitch_rate, axis=0)
    last_acceleration = pitch_acceleration[-1][None, :]
    pitch_acceleration = np.concatenate((pitch_acceleration, last_acceleration), axis=0)[:, None, :] 
    new_state = np.concatenate((states[:, :6, :], pitch_acceleration, states[:, 6, :][:, None, :]), axis=1)
    return new_state

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
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 9))
    
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
        ax = axes[i+2]
        ax.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        ax.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        ax.plot(t, percentile_50[:, i], color='r', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{state_names[i]}')
        ax.set_title(f'Time evolution of {state_names[i]}')
        ax.set_xlim(start_time, end_time)
        ax.grid(True)
    '''
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
    '''
    n_simulation = wind_speed.shape[1]
    
    plt.tight_layout()
    plt.savefig(f'./results_figure/percentile_{n_simulation}simulations_{end_time}seconds.png')
    plt.close(fig)
    
    return percentile_87_5, percentile_12_5, percentile_62_5, percentile_37_5
    
    

def plot_trajectories(t, state, wind_speed, wave_eta, seeds):
    
    figure_directory = 'results_figure_1'
    
    ######################################################################
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Pitch Acceleration (deg/s^2)', 'Rotor Speed (rpm)']
    
    safe_state_names = ['Surge', 'Surge_Velocity', 'Heave', 'Heave_Velocity', 
                   'Pitch_Angle', 'Pitch_Rate', 'Pitch_Acceleration', 'Rotor_Speed']

    # record the index of simulation that have the most occurrence of max and min
    # value for each 7 states
    max_occ_sim = []
    min_occ_sim = []
    
    # record the index of simulation that have the max and min value through all
    # time steps for each 7 states
    max_value_sim = []
    min_value_sim = []
    
    data = np.load('percentile_data/percentile_extreme.npz')

    percentile_87_5 = data['percentile_87_5']
    percentile_12_5 = data['percentile_12_5']
    percentile_62_5 = data['percentile_62_5']
    percentile_37_5 = data['percentile_37_5']
    percentile_50 = data['percentile_50']
    max_state = data['max_state']
    min_state = data['min_state']
    
    data.close()
    
    num_state = state.shape[1]
    for i in range(num_state):
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
    
    max_occ_seeds = []
    min_occ_seeds = []
    max_value_seeds = []
    min_value_seeds = []
    
    print('******************************************************************')
    print(figure_directory)
    print('******************************************************************')
    for i in range(num_state):
        max_occ_seeds.append(seeds[:, max_occ_sim[i]])
        min_occ_seeds.append(seeds[:, min_occ_sim[i]])
        max_value_seeds.append(seeds[:, max_value_sim[i]])
        min_value_seeds.append(seeds[:, min_value_sim[i]])
        
        print(f'min_occ {state_names[i]}:', min_occ_sim[i], 'seeds:', min_occ_seeds[i])
        print(f'max_occ {state_names[i]}:', max_occ_sim[i], 'seeds:', max_occ_seeds[i])    
        print(f'min_value {state_names[i]}:', min_value_sim[i], 'seeds:', min_value_seeds[i])
        print(f'max_value {state_names[i]}:', max_value_sim[i], 'seeds:', max_value_seeds[i])

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
        ax[1].set_title('Time evolution of Wave Surface Elevation at Average Surge Position')
        ax[1].set_ylabel('Wave height (m)')
        ax[1].grid(True)
        ax[1].set_xlim(0, t[-1])
        
        # plot 7 states
        for j in range(7):
            ax[j+2].plot(t, max_state[:,j], alpha=0.6, color='green', linewidth=0.5)
            ax[j+2].plot(t, min_state[:,j], alpha=0.6, color='orange', linewidth=0.5)

            ax[j+2].plot(t, state[:, j, index], color='black', linewidth=0.5)
            ax[j+2].set_xlabel('Time (s)')
            ax[j+2].set_ylabel(f'{state_names[j]}')
            
            ax[j+2].fill_between(t, percentile_12_5[:, j], percentile_87_5[:, j], color='b', alpha=0.3, edgecolor='none')
            ax[j+2].fill_between(t, percentile_37_5[:, j], percentile_62_5[:, j], color='b', alpha=0.3, edgecolor='none')
            ax[j+2].plot(t, percentile_50[:, j], color='r', alpha=0.9, linewidth=0.5)
            
            ax[j+2].set_title(f'Time evolution of {state_names[j]}')
            ax[j+2].grid(True)
            ax[j+2].set_xlim(0, t[-1])
            
        ax[9].axis('off')
        
        legend_elements = [Line2D([0], [0], color='black', lw=1, alpha=1, label='Trajectories of One Simulation Results With Extreme Events'),
                           Line2D([0], [0], color='r', lw=1, alpha=0.9, label='Median Cross All Simulations'),
                           Line2D([0], [0], color='b', lw=8, alpha=0.6, label='Central 25th Percentile of Data'),
                           Line2D([0], [0], color='b', lw=8, alpha=0.3, label='Central 75th Percentile of Data'),
                           Line2D([0], [0], color='green', lw=1, alpha=0.6, label='The Max Value Cross All Simulations at Each Time Step'),
                           Line2D([0], [0], color='orange', lw=1, alpha=0.6, label='The Min Value Cross All Simulations at Each Time Step')]
        
        ax[9].legend(handles=legend_elements, loc='center')
      
    
    # for 8 states including pitch acceleration:
    for i in range(num_state):
        # create subplots for each simulation index in max_occ_sim
        fig_max_occ, ax_max_occ = plt.subplots(5, 2, figsize=(12, 16))
        ax_max_occ = ax_max_occ.flatten()
        
        plot_helper(ax_max_occ, max_occ_sim[i])
        
        plt.tight_layout() 
        plt.savefig(f'./{figure_directory}/max_occ_{safe_state_names[i]}.png', dpi=300)
        plt.close(fig_max_occ) 
        
        
        # create subplots for each simulation index in mix_occ_sim
        fig_min_occ, ax_min_occ = plt.subplots(5, 2, figsize=(12, 16))
        ax_min_occ = ax_min_occ.flatten()
        
        plot_helper(ax_min_occ, min_occ_sim[i])
        
        plt.tight_layout() 
        plt.savefig(f'./{figure_directory}/min_occ_{safe_state_names[i]}.png', dpi=300)
        plt.close(fig_min_occ) 
        
        # create subplots for each simulation index in max_value_sim
        fig_max_value, ax_max_value = plt.subplots(5, 2, figsize=(12, 16))
        ax_max_value = ax_max_value.flatten()
        
        plot_helper(ax_max_value, max_value_sim[i])
        
        plt.tight_layout() 
        plt.savefig(f'./{figure_directory}/max_value_{safe_state_names[i]}.png', dpi=300)
        plt.close(fig_max_value) 
        
        # create subplots for each simulation index in min_value_sim
        fig_min_value, ax_min_value = plt.subplots(5, 2, figsize=(12, 16))
        ax_min_value = ax_min_value.flatten()
        
        plot_helper(ax_min_value, min_value_sim[i])
        
        plt.tight_layout() 
        plt.savefig(f'./{figure_directory}/min_value_{safe_state_names[i]}.png', dpi=300)
        plt.close(fig_min_value) 
    
def pitch_heave_extreme(state):
      
    data = np.load('percentile_data/percentile_extreme.npz')

    percentile_87_5 = data['percentile_87_5'][0]
    percentile_12_5 = data['percentile_12_5'][0]
    percentile_62_5 = data['percentile_62_5'][0]
    percentile_37_5 = data['percentile_37_5'][0]
    percentile_50 = data['percentile_50'][0]

    
    data.close()
    
    heave_percentile_87_5 = percentile_87_5[2]
    heave_percentile_12_5 = percentile_12_5[2]
    heave_percentile_62_5 = percentile_62_5[2]
    heave_percentile_37_5 = percentile_37_5[2]
    heave_percentile_50 = percentile_50 [2]
    
    pitch_percentile_87_5 = percentile_87_5[4]
    pitch_percentile_12_5 = percentile_12_5[4]
    pitch_percentile_62_5 = percentile_62_5[4]
    pitch_percentile_37_5 = percentile_37_5[4]
    pitch_percentile_50 = percentile_50 [4]
    
    
    
    max_state_indices = np.argmax(state, axis=2)
    min_state_indices = np.argmin(state, axis=2)
    
    max_heave_index = max_state_indices[:, 2]
    min_heave_index = min_state_indices[:, 2]
    
    max_pitch_index = max_state_indices[:, 4]
    min_pitch_index = min_state_indices[:, 4]
    
    heave_value = state[:,2,:]
    pitch_value = state[:,4,:]
    
    # Finding corresponding max and min values for heave and pitch using the previously found indices
    max_heave_value = heave_value[np.arange(heave_value.shape[0]), max_pitch_index]
    min_heave_value = heave_value[np.arange(heave_value.shape[0]), min_pitch_index]
    max_pitch_value = pitch_value[np.arange(pitch_value.shape[0]), max_heave_index]
    min_pitch_value = pitch_value[np.arange(pitch_value.shape[0]), min_heave_index]
    
    
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.flatten()

    ax[0].hist(max_pitch_value, bins=100, density=True, alpha=0.5, color='r', label='For Max Heave')
    ax[0].hist(min_pitch_value, bins=100, density=True, alpha=0.5, color='b', label='For Min Heave')
    
    ax[0].axvspan(pitch_percentile_12_5, pitch_percentile_87_5, color='gray', alpha=0.2, label='Central 75%')
    ax[0].axvspan(pitch_percentile_37_5, pitch_percentile_62_5, color='gray', alpha=0.4, label='Central 25%')
    ax[0].axvline(pitch_percentile_50, color='gray', alpha=0.6, linestyle='--', label='Median')
    
    ax[0].set_xlabel('Pitch (deg)')
    ax[0].set_title('Pitch Distribution at Extreme Heave')
    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[0].legend()
    
    ax[1].hist(max_heave_value, bins=100, density=True, alpha=0.5, color='r', label='For Max Pitch')
    ax[1].hist(min_heave_value, bins=100, density=True, alpha=0.5, color='b', label='For Min Pitch')
    
    ax[1].axvspan(heave_percentile_12_5, heave_percentile_87_5, color='gray', alpha=0.2, label='Central 75%')
    ax[1].axvspan(heave_percentile_37_5, heave_percentile_62_5, color='gray', alpha=0.4, label='Central 25%')
    ax[1].axvline(heave_percentile_50, color='gray', alpha=0.6, linestyle='--', label='Median')
    
    ax[1].set_xlabel('Heave (m)')
    ax[1].set_title('Heave Distribution at Extreme Pitch')
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].legend()
    
    plt.tight_layout() 
    plt.savefig('./results_figure/heave_pitch_extreme_distr.png', dpi=300)
    plt.close()
    
    
def pitchAnaly(state):
    pitch = state[:, 4, :]
    pitch_rate = state[:, 5, :]

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
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

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
    
def extremeOccurDen_distribution(state):
    
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor Speed (rpm)']
    
    max_state = np.max(state, axis=2)
    min_state = np.min(state, axis=2)
    
    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    ax = ax.flatten()
    fig.suptitle('Distribution of Extreme Values for Each State from Monte Carlo Simulation', fontsize=16, y=1)
    for i in range(7):
        
        cur_state = state[:, i, :].reshape(-1)
        
        kde_max = gaussian_kde(max_state[:,i])
        kde_min = gaussian_kde(min_state[:,i])
        kde_state = gaussian_kde(cur_state)
        
        x_max = np.linspace(min(max_state[:,i]), max(max_state[:,i]), 1000)
        x_min = np.linspace(min(min_state[:,i]), max(min_state[:,i]), 1000)
        x_state = np.linspace(min(cur_state), max(cur_state), 1000)
        
        # plot max
        ax[i].hist(max_state[:,i], bins=100, density=True, alpha=0.5, color='r', label='Max')
        ax[i].plot(x_max, kde_max(x_max), 'k', lw=1)
        
        # plot min
        ax[i].hist(min_state[:,i], bins=100, density=True, alpha=0.5, color='b', label='Min')
        ax[i].plot(x_min, kde_min(x_min), 'k', lw=1)
        
        # plot all
        ax[i].hist(cur_state, bins=100, density=True, alpha=0.5, color='gray', label='All Distribution')
        ax[i].plot(x_state, kde_state(x_state), 'k', lw=1)
        
        ax[i].set_xlabel(state_names[i])
        ax[i].set_ylabel('Density')
        ax[i].grid(True, linestyle='--', alpha=0.7)


    ax[7].axis('off')
    
    legend_elements = [Line2D([0], [0], color='r', lw=8, alpha=0.5, label='Max Value Distribution'),
                       Line2D([0], [0], color='b', lw=8, alpha=0.5, label='Min Value Distribution'),
                       Line2D([0], [0], color='gray', lw=8, alpha=0.5, label='All Data Distribution')]
    
    ax[7].legend(handles=legend_elements, loc='center')
        
    plt.tight_layout() 
    plt.savefig('./results_figure/density.png', dpi=300)
    plt.close()
    
def extremeValueDen_distribution(state):
    
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Pitch Acceleration (deg/s^2)']
    
    max_state = np.max(state, axis=0)
    min_state = np.min(state, axis=0)
    
    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    ax = ax.flatten()
    fig.suptitle('The Distribution and Extreme Value for Each State from Monte Carlo Simulation', fontsize=16, y=1)
    for i in range(7):
        
        cur_state = state[:, i, :].reshape(-1)
        
        kde_max = gaussian_kde(max_state[i])
        kde_min = gaussian_kde(min_state[i])
        kde_state = gaussian_kde(cur_state)
        
        x_max = np.linspace(min(max_state[i]), max(max_state[i]), 1000)
        x_min = np.linspace(min(min_state[i]), max(min_state[i]), 1000)
        x_state = np.linspace(min(cur_state), max(cur_state), 1000)
        
        # plot max
        ax[i].hist(max_state[i], bins=100, density=True, alpha=0.5, color='r', label='Max')
        ax[i].plot(x_max, kde_max(x_max), 'k', lw=1)
        
        # plot min
        ax[i].hist(min_state[i], bins=100, density=True, alpha=0.5, color='b', label='Min')
        ax[i].plot(x_min, kde_min(x_min), 'k', lw=1)
        
        # plot all
        ax[i].hist(cur_state, bins=100, density=True, alpha=0.5, color='gray', label='All Distribution')
        ax[i].plot(x_state, kde_state(x_state), 'k', lw=1)
        
        ax[i].set_xlabel(state_names[i])
        ax[i].set_ylabel('Density')
        ax[i].grid(True, linestyle='--', alpha=0.7)


    ax[7].axis('off')
    
    legend_elements = [Line2D([0], [0], color='r', lw=8, alpha=0.5, label='Max Value Distribution'),
                       Line2D([0], [0], color='b', lw=8, alpha=0.5, label='Min Value Distribution'),
                       Line2D([0], [0], color='gray', lw=8, alpha=0.5, label='All Data Distribution')]
    
    ax[7].legend(handles=legend_elements, loc='center')
        
    plt.tight_layout() 
    plt.savefig('./results_figure/density_value.png', dpi=300)
    plt.close()
        


    
def correl_wave_state(states, wave_eta):
    
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Pitch Acceleration (deg/s^2)']
    
    data = np.load('percentile_data/percentile_extreme.npz')

    percentile_87_5 = data['percentile_87_5']
    percentile_12_5 = data['percentile_12_5']
    percentile_62_5 = data['percentile_62_5']
    percentile_37_5 = data['percentile_37_5']
    percentile_50 = data['percentile_50']

    
    data.close()
    
    wave = wave_eta[::10].flatten('F') 
    
    nbins = 100
    
    # Bin edges
    bin_edges = np.linspace(min(wave), max(wave), nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    ax = ax.flatten()
    fig.suptitle('Correlation Between Wave Elevation and Each State', fontsize=16, y=1)
    
    for i in range(7):
        state = states[::10, i, :].flatten('F') 

        # Average values within each bin
        avg_state = []
        for j in range(nbins):
            indices = (wave >= bin_edges[i]) & (wave < bin_edges[i + 1])
            avg_state.append(np.mean(state[indices]))
            
        # Scatter plot of averages
        ax[i].scatter(bin_centers, avg_state)
        
        state_87_5 = percentile_87_5[:, i]
        state_12_5 = percentile_12_5[:, i]
        state_62_5 = percentile_62_5[:, i]
        state_37_5 = percentile_37_5[:, i]
        state_50 = percentile_50[:, i]
        
        # Filling regions for pitch and heave percentiles
        ax[i].axhspan(state_12_5[0], state_87_5[0], color='gray', alpha=0.2, label='Central 75%')
        ax[i].axhspan(state_37_5[0], state_62_5[0], color='gray', alpha=0.4, label='Central 25%')
        ax[i].axhline(state_50[0], color='gray', alpha=0.6, linestyle='--', label='Median')
        
        # Setting labels and title
        ax[i].set_ylabel(f'Average {state_names[i]}')
        ax[i].set_xlabel('Wave Elevation (m)')
        ax[i].legend()
        ax[i].grid(True)
        
    ax[7].axis('off')
    plt.tight_layout() 
    plt.savefig('./results_figure/corr_wave.png', dpi=300)
    plt.close()

    
def corr_pitch_heave(state):
    pitch = state[::8, 4, :].reshape(-1)
    heave = state[::8, 2, :].reshape(-1)
    
    nbins = 100

    # Bin edges
    bin_edges = np.linspace(min(heave), max(heave), nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Average values within each bin
    avg_pitch = []
    for i in range(nbins):
        indices = (heave >= bin_edges[i]) & (heave < bin_edges[i + 1])
        avg_pitch.append(np.mean(pitch[indices]))

    # Scatter plot of averages
    plt.scatter(bin_centers, avg_pitch)

    # Calculating and marking percentile regions
    pitch_87_5 = np.percentile(pitch, 87.5)
    pitch_12_5 = np.percentile(pitch, 12.5)
    heave_87_5 = np.percentile(heave, 87.5)
    heave_12_5 = np.percentile(heave, 12.5)
    pitch_62_5 = np.percentile(pitch, 62.5)
    pitch_37_5 = np.percentile(pitch, 37.5)
    heave_62_5 = np.percentile(heave, 62.5)
    heave_37_5 = np.percentile(heave, 37.5)
    pitch_50 = np.percentile(pitch, 50)
    heave_50 = np.percentile(heave, 50)

    # Filling regions for pitch and heave percentiles
    plt.axhspan(pitch_12_5, pitch_87_5, color='gray', alpha=0.2)
    plt.axvspan(heave_12_5, heave_87_5, color='gray', alpha=0.2, label='Central 75%')
    plt.axhspan(pitch_37_5, pitch_62_5, color='gray', alpha=0.4)
    plt.axvspan(heave_37_5, heave_62_5, color='gray', alpha=0.4, label='Central 25%')
    plt.axhline(pitch_50, color='gray', alpha=0.6, linestyle='--')
    plt.axvline(heave_50, color='gray', alpha=0.6, label='Median')

    # Setting labels and title
    plt.ylabel('Average Pitch (deg)')
    plt.xlabel('Heave (m)')
    plt.title('Binned Scatter Plot of Heave vs. Pitch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results_figure/corr_pitch_heave.png')
    plt.close()
    

def distribution(state):

    for i in range(7):
        state_20 = state[::20, i, :]
        state_final = state_20[-10:]
        all_state = state_final.reshape(-1)
        
        kde_state = gaussian_kde(all_state)
        with open(f'./density_function/state_{i}.pkl', 'wb') as f:
            pickle.dump(kde_state, f)
            
            
def save_percentile_extreme(t, state, wind_speed, wave_eta):
    
    # Get the central 75% ####################
    # States
    
    percentile_87_5 = np.nanpercentile(state, 87.5, axis=2)
    percentile_12_5 = np.nanpercentile(state, 12.5, axis=2)

    # Wind speed
    wind_percentile_87_5 = np.nanpercentile(wind_speed, 87.5, axis=1)
    wind_percentile_12_5 = np.nanpercentile(wind_speed, 12.5, axis=1)
    
    # Wave elevation
    wave_percentile_87_5 = np.nanpercentile(wave_eta, 87.5, axis=1)
    wave_percentile_12_5 = np.nanpercentile(wave_eta, 12.5, axis=1)
    
    
    # Get the central 25% ####################
    # States
    percentile_62_5 = np.nanpercentile(state, 62.5, axis=2)
    percentile_37_5 = np.nanpercentile(state, 37.5, axis=2)
    
    # Wind speed
    wind_percentile_62_5 = np.nanpercentile(wind_speed, 62.5, axis=1)
    wind_percentile_37_5 = np.nanpercentile(wind_speed, 37.5, axis=1)
    
    # Wave elevation
    wave_percentile_62_5 = np.nanpercentile(wave_eta, 62.5, axis=1)
    wave_percentile_37_5 = np.nanpercentile(wave_eta, 37.5, axis=1)

    
    # Get the median (50%) ####################
    # States
    percentile_50 = np.nanpercentile(state, 50, axis=2)
    
    # Wind speed
    wind_percentile_50 = np.nanpercentile(wind_speed, 50, axis=1)
    
    # Wave elevation
    wave_percentile_50 = np.nanpercentile(wave_eta, 50, axis=1)
    
    max_state = np.nanmax(state, axis=2)
    min_state = np.nanmin(state, axis=2)
    
    np.savez('percentile_data/percentile_extreme.npz', t=t,  
                 percentile_87_5 = percentile_87_5,
                 percentile_12_5 = percentile_12_5,
                 wind_percentile_87_5 = wind_percentile_87_5,
                 wind_percentile_12_5 = wind_percentile_12_5,
                 wave_percentile_87_5 = wave_percentile_87_5,
                 wave_percentile_12_5 = wave_percentile_12_5,
                 percentile_62_5 = percentile_62_5,
                 percentile_37_5 = percentile_37_5,
                 wind_percentile_62_5 = wind_percentile_62_5,
                 wind_percentile_37_5 = wind_percentile_37_5,
                 wave_percentile_62_5 = wave_percentile_62_5,
                 wave_percentile_37_5 = wave_percentile_37_5,
                 percentile_50 = percentile_50,
                 wind_percentile_50 = wind_percentile_50,
                 wave_percentile_50 = wave_percentile_50,
                 max_state = max_state,
                 min_state = min_state)
                 
    

    
def pitch_acceleration(state, seeds):
    '''
    this function find the extreme pitch accelaration, output the seeds
    '''
    pitch_rate = state[:, 5, :]
    path = "results_1"
    print(path)
    
    pitch_acceleration = np.diff(pitch_rate, axis=0)
    np.save(f'./pitch_acceleration/pitch_acceleration_{path}.npy', pitch_acceleration)
    
    max_acrose_simulation = np.max(pitch_acceleration, axis=1)
    min_acrose_simulation = np.min(pitch_acceleration, axis=1)
    
    # find the time step where the max and min occur
    max_value_time = np.argmax(max_acrose_simulation)
    min_value_time = np.argmin(min_acrose_simulation)
    
    # store the simulation index that have the most occurrence of max
    # and min at each time step
    max_index = np.argmax(pitch_acceleration, axis=1)
    min_index = np.argmin(pitch_acceleration, axis=1)
    
    print("max value index:", max_index[max_value_time], seeds[:, max_index[max_value_time]])
    print("min value index:", min_index[min_value_time], seeds[:, min_index[min_value_time]])
    
    # from max_index, all time steps, count the occurrence of max and min
    # for each simulation
    max_counts = np.bincount(max_index, minlength=pitch_acceleration.shape[1])
    min_counts = np.bincount(min_index, minlength=pitch_acceleration.shape[1])
    
    # for each state, find the simulation that have the most occurrence
    # of max and min value
    print("max occ index:", np.argmax(max_counts), "seeds:", seeds[:, np.argmax(max_counts)])
    print("max occ index:", np.argmax(max_counts), "seeds:", seeds[:, np.argmax(min_counts)])

def analyze_seeds(seeds):
    seeds_wind = seeds[:2]
    seedsT = seeds_wind.T
    unique_columns = {tuple(column) for column in seedsT}
    
    print(len(unique_columns))
    
    
def fft_wave(wave_eta, t):
    fft_result = np.fft.fft(wave_eta, axis=0)
    
    # Calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fft_result)
    
    # Create the corresponding frequency values for the x-axis
    frequencies = np.fft.fftfreq(len(wave_eta), t[1] - t[0])
    
    scaling_factor = 1 / len(wave_eta)
    
    amplitude_spectrum_meters = amplitude_spectrum * scaling_factor
    
    amplitude_spectrum_50 = np.median(amplitude_spectrum_meters, axis=1)
    amplitude_spectrum_87_5 = np.percentile(wave_eta, 87.5, axis=1)
    amplitude_spectrum_12_5 = np.percentile(wave_eta, 12.5, axis=1)
    amplitude_spectrum_62_5 = np.percentile(wave_eta, 62.5, axis=1)
    amplitude_spectrum_37_5 = np.percentile(wave_eta, 37.5, axis=1)
    
    np.savez('fft_data/fft_wave.npz', frequencies=frequencies,
                                      amplitude_spectrum_50 = amplitude_spectrum_50,
                                      amplitude_spectrum_87_5 = amplitude_spectrum_87_5,
                                      amplitude_spectrum_12_5 = amplitude_spectrum_12_5,
                                      amplitude_spectrum_62_5 = amplitude_spectrum_62_5,
                                      amplitude_spectrum_37_5 = amplitude_spectrum_37_5)
    
    plt.figure()
    plt.plot(frequencies, amplitude_spectrum_50, color='r', label='Medium')
    plt.fill_between(frequencies, amplitude_spectrum_12_5, amplitude_spectrum_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(frequencies, amplitude_spectrum_37_5, amplitude_spectrum_62_5, color='b', alpha=0.3, edgecolor='none')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (m)')
    plt.title('Amplitude Spectrum (FFT) of Wave Signals')
    plt.grid(True)
    plt.xlim(0.05, 0.25)
    plt.savefig('./results_figure/fft_wave.png', dpi=200)
    plt.close()
    

def largest_std(one_state, seeds):
    """
    calculate standard deviation
    
    param
    -------
    one state to calculate
    -------

    Returns 
    -------
    the seeds with the largest standard deviation

    """
    flattened_data = one_state.flatten()

    cleaned_data = flattened_data[~np.isnan(flattened_data)]

    overall_std = np.std(cleaned_data)

    
    print(f"Overall standard deviation across all simulations for surge: {overall_std}")
    
    std_devs = np.std(one_state, axis=0)
    indices_of_largest_stds = np.argsort(std_devs)[-100:]
    
    for i in indices_of_largest_stds:
        seed = seeds[:, i]
        print(f'[{seed[0]}, {seed[1]}, {seed[2]}]', "std: ", std_devs[i])
        np.save(f'large_std_pitch/{seed[0]}_{seed[1]}_{seed[2]}', one_state[:, i])
    

t, temp_state, wind_speed, wave_eta, seeds = load_data()
#state = merge_pitch_acc(temp_state)
#save_percentile_extreme(t, temp_state, wind_speed, wave_eta)
largest_std(temp_state[:, 4], seeds)



    

