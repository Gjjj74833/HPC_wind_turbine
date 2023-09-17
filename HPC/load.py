# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:16:33 2023
Load the simulation results

@author: Yihan Liu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

    # Get the central 75% ####################
    # States
    percentile_87_5 = np.percentile(state, 87.5, axis=2)
    percentile_12_5 = np.percentile(state, 12.5, axis=2)

    percentile_99_99 = np.percentile(state, 99.999, axis=2)
    percentile_0_01 = np.percentile(state, 0.001, axis=2)
    
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
        
    
    state_names = ['Surge_m', 'Surge_Velocity_m_s', 'Heave_m', 'Heave_Velocity_m_s', 
                   'Pitch_Angle_deg', 'Pitch_Rate_deg_s', 'Rotor_speed_rpm']
    
    start_time = 0
    end_time = t[-1]
    
    #if end_time > 1000:
    #   start_time = end_time - 1000
    
    now = datetime.now()
    time = now.strftime('%Y-%m-%d_%H-%M-%S')    
        
    # Plot wind speed
    plt.figure(figsize=(6.4, 2.4))
    plt.fill_between(t, wind_percentile_12_5, wind_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, wind_percentile_37_5, wind_percentile_62_5, color='b', alpha=1)
    plt.plot(t, wind_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Time evolution of Wind Speed')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Wind_Speed_{time}.png')

    
    # Plot wave_eta
    plt.figure(figsize=(6.4, 2.4))
    plt.fill_between(t, wave_percentile_12_5, wave_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, wave_percentile_37_5, wave_percentile_62_5, color='b', alpha=1)
    plt.plot(t, wave_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Water Surface Elevation at x = 0 (m)')
    plt.title('Time evolution of Wave Surface Elevation at x = 0')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Wave_Eta_{time}.png')

    
    
    # Plot all states
    for i in range(7):
        plt.figure(figsize=(6.4, 2.4))
        plt.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        plt.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        plt.plot(t, percentile_50[:, i], color='r', linewidth=1) 
        plt.plot(t, percentile_99_99[:, i], linewidth=1)
        plt.plot(t, percentile_0_01[:, i], linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(start_time, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        plt.savefig(f'./results_figure/{state_names[i]}_{time}.png')  
        

        plt.figure(figsize=(6.4, 2.4))
        plt.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        plt.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        plt.plot(t, percentile_50[:, i], color='r', linewidth=1) 
        plt.xlabel('Time (s)')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(end_time - 30, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        short = '_30s'
        plt.savefig(f'./results_figure/{safe_filename + short}_{time}.png')  

        
    # Plot average tension force on each rod
    plt.figure(figsize=(6.4, 2.4))
    plt.fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    plt.plot(t, Qt_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Averga Tension Force Per Line (N)')
    plt.title('Time evolution of Averga Tension Force Per Line')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Tension_force_{time}.png')

    
    plt.figure(figsize=(6.4, 2.4))
    plt.fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    plt.plot(t, Qt_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Averga Tension Force Per Line')
    plt.title('Time evolution of Averga Tension Force Per Line (N)')
    plt.grid(True)
    plt.xlim(end_time - 30, end_time)
    plt.savefig('./results_figure/Tension_force_30s_{time}.png')
    
    
plot_quantiles()


    

