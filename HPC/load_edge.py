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




def load_data(directory):
    
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
    white_noise_ml = [data['white_noise_ml'] for data in datas] 
    rope_tensions = [data['rope_tension'] for data in datas] 

    
    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    wind_speed = np.hstack(wind_speeds)
    wave_eta = np.hstack(wave_etas)
    seeds = np.hstack(seeds)
    white_noise_ml = np.hstack(white_noise_ml)
    rope_tension = np.concatenate(rope_tensions, axis=2)

    
    return t, state, wind_speed, wave_eta, seeds, rope_tension, white_noise_ml

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
    
def pitch_distribution(pitch, pitch_rate):
    """
    Analyzes pitch and pitch_rate by plotting their PDFs and 
    the distribution of extreme values (max and min) observed in each sample.
    
    Parameters:
    pitch (np.ndarray): 2D array for pitch where the first dimension is time and the second dimension is simulation index.
    pitch_rate (np.ndarray): 2D array for pitch_rate where the first dimension is time and the second dimension is simulation index.
    """
    
    def plot_pdf(data, ax, label):
        # Flatten the array
        flattened_data = data.flatten()

        # Calculate the PDF using KDE
        kde = gaussian_kde(flattened_data)
        x = np.linspace(flattened_data.min(), flattened_data.max(), 1000)
        pdf = kde(x)

        # Calculate the max and min values for each sample (i.e., across all simulations for each time step)
        #max_values = data.max(axis=0)
        #min_values = data.min(axis=0)

        # Calculate the PDFs of the max and min values
        #kde_max = gaussian_kde(max_values)
        #kde_min = gaussian_kde(min_values)
        #x_max = np.linspace(max_values.min(), max_values.max(), 1000)
        #x_min = np.linspace(min_values.min(), min_values.max(), 1000)
        #pdf_max = kde_max(x_max)
        #pdf_min = kde_min(x_min)

        # Plot all PDFs on the same axes
        #ax.hist(max_values, bins=100, density=True, alpha=0.5, color='r', label='Max')
        #ax.hist(min_values, bins=100, density=True, alpha=0.5, color='b', label='Min')
        ax.hist(flattened_data, bins=100, density=True, alpha=0.5, color='gray', label='All Distribution')
        ax.plot(x, pdf, color='black')
        #ax.plot(x_max, pdf_max, color='red')
        #ax.plot(x_min, pdf_min, color='blue')
        ax.set_xlabel(f'{label}')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    # Plot pitch distributions
    plot_pdf(pitch, axs[0], 'Wind Speed (m/s)')

    # Plot pitch_rate distributions
    plot_pdf(pitch_rate, axs[1], 'Wave Elevation (m)')

    plt.tight_layout()
    plt.savefig('./figure/wind_wave_distr.png')
    
def pitch_distr_compare(pitch_or, pitch_rate_or, pitch_lo, pitch_rate_lo):
    """
    Compare pitch distribution for important sampling.
    
    Parameters:
    pitch_or (np.ndarray): 2D array for original pitch where the first dimension is time and the second dimension is simulation index.
    pitch_rate_or (np.ndarray): 2D array for original pitch rate where the first dimension is time and the second dimension is simulation index.
    pitch_lo (np.ndarray): 2D array for pitch from importance sampling where the first dimension is time and the second dimension is simulation index.
    pitch_rate_lo (np.ndarray): 2D array for pitch rate from importance sampling where the first dimension is time and the second dimension is simulation index.
    """
    
    def plot_pdf(data_or, data_lo, ax, xlabel, ylabel):
        # Flatten the arrays
        flattened_data_or = data_or.flatten()
        flattened_data_lo = data_lo.flatten()

        # Calculate the PDFs using KDE
        kde_or = gaussian_kde(flattened_data_or)
        kde_lo = gaussian_kde(flattened_data_lo)
        x_or = np.linspace(flattened_data_or.min(), flattened_data_or.max(), 1000)
        x_lo = np.linspace(flattened_data_lo.min(), flattened_data_lo.max(), 1000)
        pdf_or = kde_or(x_or)
        pdf_lo = kde_lo(x_lo)

        # Calculate the max and min values for each sample
        max_values_or = data_or.max(axis=0)
        min_values_or = data_or.min(axis=0)
        max_values_lo = data_lo.max(axis=0)
        min_values_lo = data_lo.min(axis=0)

        # Calculate the PDFs of the max and min values
        kde_max_or = gaussian_kde(max_values_or)
        kde_min_or = gaussian_kde(min_values_or)
        kde_max_lo = gaussian_kde(max_values_lo)
        kde_min_lo = gaussian_kde(min_values_lo)
        x_max_or = np.linspace(max_values_or.min(), max_values_or.max(), 1000)
        x_min_or = np.linspace(min_values_or.min(), min_values_or.max(), 1000)
        x_max_lo = np.linspace(max_values_lo.min(), max_values_lo.max(), 1000)
        x_min_lo = np.linspace(min_values_lo.min(), min_values_lo.max(), 1000)
        pdf_max_or = kde_max_or(x_max_or)
        pdf_min_or = kde_min_or(x_min_or)
        pdf_max_lo = kde_max_lo(x_max_lo)
        pdf_min_lo = kde_min_lo(x_min_lo)

        # Plot all PDFs on the same axes
        ax.hist(max_values_or, bins=50, density=True, alpha=0.5, color='gray', label='Original MCMC Distribution', histtype='stepfilled')
        ax.hist(min_values_or, bins=50, density=True, alpha=0.5, color='gray', histtype='stepfilled')
        ax.hist(flattened_data_or, bins=50, density=True, alpha=0.5, color='gray', histtype='stepfilled')
        ax.plot(x_or, pdf_or, color='black')
        ax.plot(x_max_or, pdf_max_or, color='black')
        ax.plot(x_min_or, pdf_min_or, color='black')

        ax.hist(max_values_lo, bins=50, density=True, alpha=0.5, color='red', label='Importance Sampling Distribution', histtype='stepfilled')
        ax.hist(min_values_lo, bins=50, density=True, alpha=0.5, color='red', histtype='stepfilled')
        ax.hist(flattened_data_lo, bins=50, density=True, alpha=0.5, color='red', histtype='stepfilled')
        ax.plot(x_lo, pdf_lo, color='red')
        ax.plot(x_max_lo, pdf_max_lo, color='red')
        ax.plot(x_min_lo, pdf_min_lo, color='red')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    # Plot pitch distributions
    plot_pdf(pitch_or, pitch_lo, axs[0], 'Pitch (deg)', 'Density')

    # Plot pitch_rate distributions
    plot_pdf(pitch_rate_or, pitch_rate_lo, axs[1], 'Pitch rate (deg/s)', 'Density')

    plt.tight_layout()
    plt.savefig('./figure/pitch_distr_compare.png')
    

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
    
def largest_std_percentage(one_state, seeds, threshold, file_name):
    """
    Calculate standard deviation and print seeds with standard deviation greater than a threshold.

    Parameters
    ----------
    one_state : np.ndarray
        The state array to calculate standard deviation for.
    seeds : np.ndarray
        The array of seeds corresponding to each simulation.
    threshold : float
        The threshold for standard deviation.

    Returns
    -------
    None
    """
    # Flatten the data and remove NaN values
    flattened_data = one_state.flatten()
    cleaned_data = flattened_data[~np.isnan(flattened_data)]

    # Calculate the overall standard deviation
    overall_std = np.std(cleaned_data)
    #print(f"Overall standard deviation across all simulations: {overall_std}")

    # Calculate standard deviations along the second axis
    std_devs = np.std(one_state, axis=0)

    # Find indices of standard deviations greater than the threshold
    indices_of_large_stds = np.where(std_devs > threshold)[0]

    # Sort indices by standard deviation in descending order
    sorted_indices = indices_of_large_stds[np.argsort(std_devs[indices_of_large_stds])[::-1]]

    # Print the count of samples with std greater than the threshold
    count_above_threshold = len(indices_of_large_stds)
    
    with open(file_name, 'w') as file:
        file.write(f"Overall standard deviation across all simulations: {overall_std}\n")
        file.write(f"Number of samples with standard deviation greater than {threshold}: {count_above_threshold}\n")
        for i in sorted_indices:
            seed = seeds[:, i]
            file.write(f'[{seed[0]}, {seed[1]}, {seed[2]}] std: {std_devs[i]}\n')
        
    #print(f"Number of samples with standard deviation greater than {threshold}: {count_above_threshold}")

    # Print the seeds and their standard deviations
    #for i in sorted_indices:
    #    seed = seeds[:, i]
    #    print(f'[{seed[0]}, {seed[1]}, {seed[2]}] std: {std_devs[i]}') 
    
'''
def extract_extreme(state, seeds, upper_bound, lower_bound, config_ID, epsilon):
    
    """
    count number of samples exceed threshold
    """
    
    
    count = 0
    
    for i in range(state.shape[1]):
        max_value = np.max(state[:, i])
        min_value = np.min(state[:, i])
        
        if max_value > upper_bound or min_value < lower_bound:
            seed = seeds[:, i]
            print(f'[{seed[0]}, {seed[1]}, {seed[2]}], max = {max_value}, min = {min_value}')
            count += 1
        
    print(f'Extreme events exceed {upper_bound} for Configuration {config_ID}, epsilon={epsilon}: {count}, percentage: {count/state.shape[1]}')
        
'''
def extract_extreme(state, upper_bound, epsilon):
    """
    Count the number of samples that exceed the upper bound.
    """
    count = 0
    index = []
    for i in range(state.shape[1]):
        max_value = np.max(state[:, i])
        
        if max_value > upper_bound:
            #seed = seeds[:, i]
            #print(f'[{seed[0]}, {seed[1]}, {seed[2]}], max = {max_value} exceeds upper bound {upper_bound}')
            count += 1
            index.append(i)
        
    print(f'Extreme events exceed {upper_bound} for {state.shape[1]} samples, epsilon={epsilon}: {count}, percentage: {count/state.shape[1]:.2%}')
    return index
    
def compare_PDFs(states, state_labels, name, unit):
    """
    Compare the distribution of multiple states in a single figure.
    
    Parameters:
    states (list of np.ndarray): List of 2D arrays where each array is a state.
    state_labels (list of str): List of labels for each state, corresponding to the state arrays.
    name (str): used for save figure name
    unit (str): state name with unit for x label
    """
    
    def plot_pdf(data, ax, label, is_first=False):
        # Flatten the array
        flattened_data = data.flatten()

        # Calculate the PDFs using KDE
        kde = gaussian_kde(flattened_data)
        x = np.linspace(flattened_data.min(), flattened_data.max(), 1000)
        pdf = kde(x)

        # Plot the PDFs
        if is_first:
            ax.plot(x, pdf, label=label, linestyle='-', color='black', linewidth=2.5)
        else:
            ax.plot(x, pdf, label=label, linestyle='-', linewidth=1.5)

        ax.set_xlabel(unit)
        ax.set_ylabel('Density')
        ax.grid(True)

    # Create a figure for the combined plots
    fig, ax = plt.subplots(figsize=(6, 3))

    # Iterate through all states and plot their PDFs on the same axes
    for i, (data, label) in enumerate(zip(states, state_labels)):
        plot_pdf(data, ax, label, is_first=(i == 0))

    # Adjust the legend to be outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
    plt.savefig(f'./figure/{name}_pdf_compare.png', bbox_inches='tight')
    
    
def state_PDF_compare(state_or, state_lo):
    """
    Compare pitch distribution for important sampling.
    
    Parameters:
    pitch_or (np.ndarray): 2D array for original pitch where the first dimension is time and the second dimension is simulation index.
    pitch_rate_or (np.ndarray): 2D array for original pitch rate where the first dimension is time and the second dimension is simulation index.
    pitch_lo (np.ndarray): 2D array for pitch from importance sampling where the first dimension is time and the second dimension is simulation index.
    pitch_rate_lo (np.ndarray): 2D array for pitch rate from importance sampling where the first dimension is time and the second dimension is simulation index.
    """
    
    def plot_pdf(data_or, data_lo, ax, xlabel, ylabel):
        # Flatten the arrays
        flattened_data_or = data_or.flatten()
        flattened_data_lo = data_lo.flatten()

        # Calculate the PDFs using KDE
        kde_or = gaussian_kde(flattened_data_or)
        kde_lo = gaussian_kde(flattened_data_lo)
        x_or = np.linspace(flattened_data_or.min(), flattened_data_or.max(), 1000)
        x_lo = np.linspace(flattened_data_lo.min(), flattened_data_lo.max(), 1000)
        pdf_or = kde_or(x_or)
        pdf_lo = kde_lo(x_lo)

        # Calculate the max and min values for each sample
        max_values_or = data_or.max(axis=0)
        min_values_or = data_or.min(axis=0)
        max_values_lo = data_lo.max(axis=0)
        min_values_lo = data_lo.min(axis=0)

        # Calculate the PDFs of the max and min values
        kde_max_or = gaussian_kde(max_values_or)
        kde_min_or = gaussian_kde(min_values_or)
        kde_max_lo = gaussian_kde(max_values_lo)
        kde_min_lo = gaussian_kde(min_values_lo)
        x_max_or = np.linspace(max_values_or.min(), max_values_or.max(), 1000)
        x_min_or = np.linspace(min_values_or.min(), min_values_or.max(), 1000)
        x_max_lo = np.linspace(max_values_lo.min(), max_values_lo.max(), 1000)
        x_min_lo = np.linspace(min_values_lo.min(), min_values_lo.max(), 1000)
        pdf_max_or = kde_max_or(x_max_or)
        pdf_min_or = kde_min_or(x_min_or)
        pdf_max_lo = kde_max_lo(x_max_lo)
        pdf_min_lo = kde_min_lo(x_min_lo)

        # Plot all PDFs on the same axes
        ax.hist(max_values_or, bins=50, density=True, alpha=0.5, color='gray', label='Original MCMC Distribution', histtype='stepfilled')
        ax.hist(min_values_or, bins=50, density=True, alpha=0.5, color='gray', histtype='stepfilled')
        ax.hist(flattened_data_or, bins=50, density=True, alpha=0.5, color='gray', histtype='stepfilled')
        ax.plot(x_or, pdf_or, color='black')
        ax.plot(x_max_or, pdf_max_or, color='black')
        ax.plot(x_min_or, pdf_min_or, color='black')

        ax.hist(max_values_lo, bins=50, density=True, alpha=0.5, color='red', label='Importance Sampling Distribution', histtype='stepfilled')
        ax.hist(min_values_lo, bins=50, density=True, alpha=0.5, color='red', histtype='stepfilled')
        ax.hist(flattened_data_lo, bins=50, density=True, alpha=0.5, color='red', histtype='stepfilled')
        ax.plot(x_lo, pdf_lo, color='red')
        ax.plot(x_max_lo, pdf_max_lo, color='red')
        ax.plot(x_min_lo, pdf_min_lo, color='red')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    # Create a figure with two subplots
    fig, axs = plt.subplots(figsize=(5, 3))

    # Plot pitch distributions
    plot_pdf(state_or, state_lo, axs, 'Surge (m)', 'Density')


    plt.tight_layout()
    plt.savefig('./figure/surge_distr_compare_pi0.png')
    
    
def extract_top15_saveConfig(state, white_noise_list, seeds, save_path, n=15):
    """
    Extract the samples with the top N largest values, and save the corresponding white noise
    """
    
    # Get the maximum value in each sample (column)
    max_values = np.max(state, axis=0)
    
    # Get the indices of the top N largest values, largest first
    top_indices = np.argsort(max_values)[-n:][::-1]
    
    print(f'Top {n} extreme events')
    for i in top_indices:
        max_value = max_values[i]
        seed = seeds[:, i]
        print(f'Seed: [{seed[0]}, {seed[1]}, {seed[2]}], max value = {max_value}')
        
    # Extract the corresponding white noise using top_indices
    top_white_noise = white_noise_list[:, top_indices]
    
    # Save the top N white noise samples
    np.save(save_path, top_white_noise)
    print(f'White noise for top {n} extreme events saved to {save_path}')
    print(top_white_noise.shape)
    
def largest_rope_tension(rope_tension, seeds, white_noise_ml, save_path, n=15):
    """
    Display the top n largest rope tension events along with their locations and corresponding seeds.

    Parameters:
    ----------
    rope_tension : np.array
        Array of shape (time_steps, 3, simulation_index) with tension forces for each component.
    seeds : np.array
        Array of shape (3, simulation_index) with seed information for each simulation.
    n : int
        The number of top largest tension events to display.
    """
    
    # Find the maximum tension for each time step and simulation, along with the location
    large_tension = np.max(rope_tension, axis=1)  # Shape: (time_steps, simulation_index)
    large_tension_loc = np.argmax(rope_tension, axis=1)  # Shape: (time_steps, simulation_index)

    # Get the largest tension per sample (across all time steps for each simulation)
    large_tension_per_sample = np.max(large_tension, axis=0)  # Shape: (simulation_index,)
    top_indices = np.argsort(large_tension_per_sample)[-n:][::-1]  # Top n indices in descending order

    # Calculate overall mean and standard deviation
    mean_tension = np.mean(rope_tension)
    std_tension = np.std(rope_tension)
    
    # Print results
    print(f"Overall Rope Tension Mean: {mean_tension}")
    print(f"Overall Rope Tension Standard Deviation: {std_tension}")
    print(f"Top {n} extreme events:")
    for i in top_indices:
        tension = large_tension_per_sample[i]
        max_time_step = np.argmax(large_tension[:, i])  # Time step of max tension for this simulation
        rope_loc = large_tension_loc[max_time_step, i] + 1  # Add 1 to convert to 1, 2, or 3
        seed = seeds[:, i]  # Seed for this simulation

        print(f"Seed: [{seed[0]}, {seed[1]}, {seed[2]}], Tension = {tension}, Location = {rope_loc} (Component {rope_loc})")
    
    top_white_noise = white_noise_ml[:, top_indices]
    np.save(save_path, top_white_noise)
    print(f'White noise for top {n} extreme events saved to {save_path}')
    print(top_white_noise.shape)
    #return top_indices

def calculate_3sigma_range(rope_tension):
    """
    Calculate the 3σ range for windward and leeward tensions in the rope_tension data.
    
    Parameters:
    ----------
    rope_tension : np.array
        Array of shape (time_steps, 3, simulation_index) with tension forces for each component:
        [windward, leeward, middle].
    
    Returns:
    -------
    dict
        A dictionary with the 3σ range for windward and leeward tensions.
    """
    # Extract windward and leeward tensions
    windward_tension = rope_tension[:, 0, :].flatten()  # Flatten to 1D for all time steps and simulations
    leeward_tension = rope_tension[:, 1, :].flatten()

    # Calculate mean and standard deviation
    windward_mean = np.mean(windward_tension)
    windward_std = np.std(windward_tension)
    leeward_mean = np.mean(leeward_tension)
    leeward_std = np.std(leeward_tension)

    # Calculate 3σ range
    windward_3sigma = ((windward_mean - 3 * windward_std)/2000, (windward_mean + 3 * windward_std)/2000)
    leeward_3sigma = ((leeward_mean - 3 * leeward_std)/2000, (leeward_mean + 3 * leeward_std)/2000)

    # Print results
    print(f"Windward 3σ range: {windward_3sigma} kN")
    print(f"Leeward 3σ range: {leeward_3sigma} kN")


def event_indicator_surge(state, threshold):
    """
    Return 1 if extreme event is detacted. When surge exceed the threshold

    Parameters
    ----------
    state : numpy.array
        Time serie for state, one sample
    threshold : float
        threshold

    Returns
    -------
    1 if extreme event is detacted, else return 0

    """
    return np.any(state > threshold)

def compute_R(state, white_noise_ml, epsilon, threshold):
    """
    Compute the true probability with Importance Sampling weight
    This is for single configuration
    Parameters
    ----------
    state : numpy.array
        MCMC results for one state. Shape: (time_step, sample_index)
    white_noise_ml : numpy.array
        Random phase correspond to each sample. Shape: (random_phase, sample_index)
    epsilon : std
        

    Returns
    -------
    the true probability

    """
    weight_sum = 0
    event_count = 0
    for i in range(state.shape[1]):
        if np.any(state[:, i] > threshold): # event is detacted
            event_count += 1
            phase_length = 31 # there are 31 phases, this is fixed
            # find the original sampling source
            sample_seed = 4021713
            state_before = np.random.get_state()
            np.random.seed(sample_seed)
            sampling_source = np.random.uniform(-np.pi, np.pi, phase_length) 
            np.random.set_state(state_before)
            # for each random phase, Compute IS weight and find the product
            #IS_weight = 1
            IS_weight = 0
            for phase in range(phase_length): 
                exponent = - ((white_noise_ml[phase, i] - sampling_source[phase]) ** 2) / (2 * epsilon ** 2)
                normal_density = (1 / (epsilon * np.sqrt(2 * np.pi))) * np.exp(exponent)
                uniform_density = 1 / (2 * np.pi)
                weight = uniform_density / normal_density
                
                #print("      weight = {uniform_density}/{normal_density} = {weight}")
                
                IS_weight *= weight
                #IS_weight += weight
            IS_weight = IS_weight**(1/31)
            #print("Weight for this sample is {IS_weight}")
            weight_sum += IS_weight
            
    #print("Total weight is {weight_sum}")
    
    True_exp = weight_sum / state.shape[1]
    print(f"{event_count} events detacted, percentatge = {event_count/state.shape[1]}. The true probability for threshold exceed {threshold}m, epsilon={epsilon}, for {state.shape[1]} samples is {True_exp}")
            

def save_large_phase_pdf(index, white_noise_ml, iteration):
    """
    Saves the white noise values for the specified simulation indices.

    Parameters:
        index (list): List of simulation indices to extract and save.
        white_noise_ml (np.ndarray): A numpy array with shape [31, simulation_index].
    """
    extracted_data = white_noise_ml[:, index]
    np.save(f"large_noise/extreme_large_noise_MCMC_ite{iteration}.npy", extracted_data)
    '''
    output_dir = f"large_random_phase_kde/iteration_{iteration}"
    os.makedirs(output_dir, exist_ok=True)

    
    # Save the extracted data to a file
    for i in range(extracted_data.shape[0]):

        # KDE for PDF estimation
        kde = gaussian_kde(extracted_data[i])
        save_path = f"{output_dir}/kde_{i}.pkl" 
        with open(save_path, "wb") as f:
            pickle.dump(kde, f)
    '''


t, state, wind_speed, wave_eta, seeds, rope_tension, white_noise_ml = load_data("results_adaptive_imps_surge_ite1")
index = extract_extreme(state[:, 0], 8, 0)
save_large_phase_pdf(index, white_noise_ml, 1)
    
#largest_rope_tension(rope_tension, seeds, white_noise_ml, "imps_ite/imps_tension_ml_pi0_ite1.npy")

#calculate_3sigma_range(rope_tension)
#t, state, wind_speed, wave_eta, seeds = load_data('results')
#print("state shape: ", state.shape)
#print("seeds shape: ", seeds.shape)
#extract_extreme(state[:1800, 0], seeds, 10, 0, 0)


#state = merge_pitch_acc(temp_state)
#save_percentile_extreme(t[1000:], temp_state[1000:], wind_speed[1000:], wave_eta[1000:])
#pitch_distribution(wind_speed[:-1000], wave_eta[:-1000])
#largest_std(state_2500[:, 0][:500], seeds)
#largest_std_percentage(temp_state[:, 4], seeds, 0.3259)

#pitch_distr_compare(state_original[:, 4][1000:], state_original[:, 5][1000:], state_2500[:, 4], state_2500[:, 5])

#for sample_ID in range(1, 6):
#    for elipse in [1, 2, 4, 6, 8, 0]:
#        t, state, wind_speed, wave_eta, seeds = load_data(f'results_surge_{sample_ID}_pi{elipse}')
#        largest_std_percentage(state, seeds, 0.3259, f'pitch_compare_{sample_ID}_pi{elipse}.txt')
#t, state, wind_speed, wave_eta, seeds, white_noise_ml = load_data('results_surge_n15_pi0_ite0_conv')
#extract_top15_saveConfig(state[:, 0, :5000], white_noise_ml, seeds, "imps_ite/imps_surge_ml_pi0_ite0.npy", n=15)


'''
#save selected samples
top_indices = largest_rope_tension(rope_tension, seeds)
top_state = state[:, :, top_indices]
top_wind_speed = wind_speed[:, top_indices]
top_wave_eta = wave_eta[:, top_indices]
top_seeds = seeds[:, top_indices]
top_rope_tension = rope_tension[:, :, top_indices]
np.savez("top_tension.npz", t=t, state=top_state, wind_speed=top_wind_speed, 
             wave_eta=top_wave_eta, seeds=top_seeds, rope_tension=top_rope_tension)

#convergence test for iteration
state = load_data('results_surge_n15_pi0_ite0_conv')[1]

for count in [2500, 3000, 4000, 5000, 7500, 10000, 15000, 20000]:
    print("For", count)
    for i in range(20000//count):
        extract_extreme(state[:, 0, i*count:count*(i+1)], 8, 0, 0)
        extract_extreme(state[:, 0, i*count:count*(i+1)], 9, 0, 0)
        extract_extreme(state[:, 0, i*count:count*(i+1)], 10, 0, 0)
        extract_extreme(state[:, 0, i*count:count*(i+1)], 11, 0, 0)
        extract_extreme(state[:, 0, i*count:count*(i+1)], 12, 0, 0)
        extract_extreme(state[:, 0, i*count:count*(i+1)], 13, 0, 0)
        print()
    print("____________________________________________________")
    print()


#print output for iterations
print("Standard MCMC (iteration 0)")
state = load_data('results')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))
print()


print("Iteration 1")
state = load_data('results_surge_n15_pi0_ite0')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 1")
state = load_data('results_surge_n15_pi0_ite1')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 2")
state = load_data('results_surge_n15_pi0_ite2')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 3")
state = load_data('results_surge_n15_pi0_ite3')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))


print("Iteration 4")
state = load_data('results_surge_n15_pi0_ite4')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 5")
state = load_data('results_surge_n15_pi0_ite5')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 6")
state = load_data('results_surge_n15_pi0_ite6')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 7")
state = load_data('results_surge_n15_pi0_ite7')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 8")
state = load_data('results_surge_n15_pi0_ite8')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 9")
state = load_data('results_surge_n15_pi0_ite9')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 10")
state = load_data('results_surge_n15_pi0_ite10')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))

print("Iteration 11")
state = load_data('results_surge_n15_pi0_ite11')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))


print("Iteration 12")
state = load_data('results_surge_n15_pi0_ite12')[1]
extract_extreme(state[:, 0], 8, 0)
extract_extreme(state[:, 0], 9, 0)
extract_extreme(state[:, 0], 10, 0)
extract_extreme(state[:, 0], 11, 0)
extract_extreme(state[:, 0], 12, 0)
extract_extreme(state[:, 0], 13, 0)
print("The largest value observed is:", np.max(state[:, 0]))



# plot wind pdf for n=15 at different epsilon
wind = load_data('results')[2]
wind_pi0 = load_data('results_surge_n15_pi0')[2]
wind_pi8 = load_data('results_surge_n15_pi8')[2]
wind_pi6 = load_data('results_surge_n15_pi6')[2]
wind_pi4 = load_data('results_surge_n15_pi4')[2]
wind_pi2 = load_data('results_surge_n15_pi2')[2]
wind_pi1 = load_data('results_surge_n15_pi1')[2]
compare_PDFs([wind,
              wind_pi0,
              wind_pi8,
              wind_pi6,
              wind_pi4,
              wind_pi2,
              wind_pi1], 
             ["standard MCMC",
              r"$\epsilon = 0$",
              r"$\epsilon = \pi/8$",
              r"$\epsilon = \pi/6$",
              r"$\epsilon = \pi/4$",
              r"$\epsilon = \pi/2$",
              r"$\epsilon = \pi$"], "epsilon_surge_n15", "Wind Speed (m/s)")

seeds=1
#Analysis n=15 imps
for epsilon in (1, 2, 4, 6, 8, 0):
    state = load_data(f'results_surge_n15_pi{epsilon}')[1]
    print('For epsilon =', epsilon)
    extract_extreme(state[:, 0], seeds, 8, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 8.5, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 9, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 9.5, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 10, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 10.5, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 11, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 11.5, 15, epsilon)
    extract_extreme(state[:, 0], seeds, 12, 15, epsilon)
    print()
    print()



#Extract surge events happening before 800s
for config_ID in range(1, 6):
    for epsilon in (1, 2, 4, 6, 8, 0):
        t, state, wind_speed, wave_eta, seeds = load_data(f'results_surge_{config_ID}_pi{epsilon}')
        extract_extreme(state[:1600, 0], seeds, 10, -100, config_ID, epsilon)
        print()


#compare epsilon for surge events
for config_ID in range(1, 6):
    for epsilon in [1, 2, 4, 6, 8, 0]:
        t, state, wind_speed, wave_eta, seeds = load_data(f'results_surge_{config_ID}_pi{epsilon}')
        extract_extreme(state[:,0], seeds, 10, -100, config_ID, epsilon)
        extract_extreme(state[:,0], seeds, 9, -100, config_ID, epsilon)
        extract_extreme(state[:,0], seeds, 8, -100, config_ID, epsilon)


#conpare configuration 3
state_original = load_data('results')[1]
state_PDF_compare(state_original[:, 0][1000:], load_data('results_surge_3_pi0')[1][:, 0])


#compare results for surge of all configurations with epsilon = 0
state_2 = load_data('results_surge_2_pi0')[1][:,0]
state_3 = load_data('results_surge_3_pi0')[1][:,0]
state_4 = load_data('results_surge_4_pi0')[1][:,0]
state_5 = load_data('results_surge_5_pi0')[1][:,0]

print('ID=2')
extract_extreme(state_2, seeds, 8, -100)
extract_extreme(state_2, seeds, 9, -100)
extract_extreme(state_2, seeds, 10, -100)

print('ID=3')
extract_extreme(state_3, seeds, 8, -100)
extract_extreme(state_3, seeds, 9, -100)
extract_extreme(state_3, seeds, 10, -100)

print('ID=4')
extract_extreme(state_4, seeds, 8, -100)
extract_extreme(state_4, seeds, 9, -100)
extract_extreme(state_4, seeds, 10, -100)

print('ID=5')
extract_extreme(state_5, seeds, 8, -100)
extract_extreme(state_5, seeds, 9, -100)
extract_extreme(state_5, seeds, 10, -100)


#plot pdfs for surge, compare imps and mcmc
state_original = load_data('results')[1]

state_PDF_compare(state_original[:, 0][1000:], np.hstack((load_data('results_surge_1_pi0')[1][:, 0, 565:3065],
                                           load_data('results_surge_2_pi0')[1][:, 0],
                                           load_data('results_surge_3_pi0')[1][:, 0],
                                           load_data('results_surge_4_pi0')[1][:, 0],
                                           load_data('results_surge_5_pi0')[1][:, 0])))

#convergence test
extract_extreme(state[:, 0], seeds, 10, -100)
extract_extreme(state[:, 0], seeds, 9, -100)
extract_extreme(state[:, 0], seeds, 8, -100)

extract_extreme(state[:, 0, 4589:14589], seeds, 10, -100)
extract_extreme(state[:, 0, 4589:14589], seeds, 9, -100)
extract_extreme(state[:, 0, 4589:14589], seeds, 8, -100)

extract_extreme(state[:, 0, 2321:7321], seeds, 10, -100)
extract_extreme(state[:, 0, 2321:7321], seeds, 9, -100)
extract_extreme(state[:, 0, 2321:7321], seeds, 8, -100)

extract_extreme(state[:, 0, 9784:12284], seeds, 10, -100)
extract_extreme(state[:, 0, 9784:12284], seeds, 9, -100)
extract_extreme(state[:, 0, 9784:12284], seeds, 8, -100)

extract_extreme(state[:, 0, 11685:13185], seeds, 10, -100)
extract_extreme(state[:, 0, 11685:13185], seeds, 9, -100)
extract_extreme(state[:, 0, 11685:13185], seeds, 8, -100)


#plot pdfs for wind of different epsilon average all configurations

wind_pi0 = np.hstack((load_data('results_2500')[2],
                      load_data('results_pitch_lo_1')[2],
                      load_data('results_pitch_lo_2')[2],
                      load_data('results_pitch_lo_3')[2],
                      load_data('results_pitch_lo_4')[2]))

wind_pi1 = np.hstack((load_data('results_pitch_1_pi1')[2],
                      load_data('results_pitch_2_pi1')[2],
                      load_data('results_pitch_3_pi1')[2],
                      load_data('results_pitch_4_pi1')[2],
                      load_data('results_pitch_5_pi1')[2],))

wind_pi2 = np.hstack((load_data('results_pitch_1_pi2')[2],
                      load_data('results_pitch_2_pi2')[2],
                      load_data('results_pitch_3_pi2')[2],
                      load_data('results_pitch_4_pi2')[2],
                      load_data('results_pitch_5_pi2')[2],))

wind_pi4 = np.hstack((load_data('results_pitch_1_pi4')[2],
                      load_data('results_pitch_2_pi4')[2],
                      load_data('results_pitch_3_pi4')[2],
                      load_data('results_pitch_4_pi4')[2],
                      load_data('results_pitch_5_pi4')[2],))

wind_pi6 = np.hstack((load_data('results_pitch_1_pi6')[2],
                      load_data('results_pitch_2_pi6')[2],
                      load_data('results_pitch_3_pi6')[2],
                      load_data('results_pitch_4_pi6')[2],
                      load_data('results_pitch_5_pi6')[2],))

wind_pi8 = np.hstack((load_data('results_pitch_1_pi8')[2],
                      load_data('results_pitch_2_pi8')[2],
                      load_data('results_pitch_3_pi8')[2],
                      load_data('results_pitch_4_pi8')[2],
                      load_data('results_pitch_5_pi8')[2],))

    #just see config 2
wind_pi0 = load_data('results_surge_2_pi0')[2]
wind_pi1 = load_data('results_surge_2_pi1')[2]
wind_pi2 = load_data('results_surge_2_pi2')[2]
wind_pi4 = load_data('results_surge_2_pi4')[2]
wind_pi6 = load_data('results_surge_2_pi6')[2]
wind_pi8 = load_data('results_surge_2_pi8')[2]

wind_normal = load_data('results')[2]
compare_PDFs([wind_normal,
              wind_pi0,
              wind_pi1,
              wind_pi2,
              wind_pi4,
              wind_pi6,
              wind_pi8],
             ["standard MCMC",
              r"$\epsilon = 0$",
              r"$\epsilon = \pi$",
              r"$\epsilon = \pi/2$",
              r"$\epsilon = \pi/4$",
              r"$\epsilon = \pi/6$",
              r"$\epsilon = \pi/8$"], "epsilon_surge_config2", "Wind Speed (m/s)")


#plot pdfs for different configurations with epsilon=0
wind_normal = load_data('results')[2]
wind_surge_1 = load_data("results_surge_1_pi0")[2]
wind_surge_2 = load_data("results_surge_2_pi0")[2]
wind_surge_3 = load_data("results_surge_3_pi0")[2]
wind_surge_4 = load_data("results_surge_4_pi0")[2]
wind_surge_5 = load_data("results_surge_5_pi0")[2]

compare_PDFs([wind_normal,
              wind_surge_1,
              wind_surge_2,
              wind_surge_3,
              wind_surge_4,
              wind_surge_5], ["standard MCMC",
                              "ID = 1",
                              "ID = 2",
                              "ID = 3",
                              "ID = 4",
                              "ID = 5"], "wind_surge", "Wind Speed (m/s)")


'''