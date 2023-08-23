# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:16:33 2023
Load the simulation results

@author: Yihan Liu
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_quantiles():
    
    file_path = "./results/results.npz"
    data = np.load(file_path)        

    # Access the arrays using the keys provided during saving
    state = data['state']
    wind_speed = data['wind_speed]
    wave_eta = wave_eta
    Q_t = data['Q_t']
    
    
    state_names = ['Surge_m', 'Surge_Velocity_m_s', 'Heave_m', 'Heave_Velocity_m_s', 
                   'Pitch_Angle_deg', 'Pitch_Rate_deg_s', 'Rotor_speed_rpm']
    
    start_time = 0
    end_time = t[-1]
    
    if end_time > 1000:
        start_time = end_time - 1000
        
    # Plot wind speed
    plt.figure(figsize=(12.8, 4.8))
    plt.fill_between(t, wind_percentile_12_5, wind_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, wind_percentile_37_5, wind_percentile_62_5, color='b', alpha=1)
    plt.plot(t, wind_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Time evolution of Wind Speed')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Wind_Speed.png')

    
    # Plot wave_eta
    plt.figure(figsize=(12.8, 4.8))
    plt.fill_between(t, wave_percentile_12_5, wave_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, wave_percentile_37_5, wave_percentile_62_5, color='b', alpha=1)
    plt.plot(t, wave_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Water Surface Elevation at x = 0 (m)')
    plt.title('Time evolution of Wave Surface Elevation at x = 0')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Wave_Eta.png')

    
    
    # Plot all states
    for i in range(7):
        plt.figure(figsize=(12.8, 4.8))
        plt.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3, edgecolor='none')
        plt.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        plt.plot(t, percentile_50[:, i], color='r', linewidth=1) 
        plt.xlabel('Time (s)')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(start_time, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        plt.savefig(f'./results_figure/{state_names[i]}.png')  

        
        plt.figure(figsize=(12.8, 4.8))
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
        plt.savefig(f'./results_figure/{safe_filename + short}.png')  

        
    # Plot average tension force on each rod
    plt.figure(figsize=(12.8, 4.8))
    plt.fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    plt.plot(t, Qt_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Averga Tension Force Per Line (N)')
    plt.title('Time evolution of Averga Tension Force Per Line')
    plt.grid(True)
    plt.xlim(start_time, end_time)
    plt.savefig('./results_figure/Tension_force.png')

    
    plt.figure(figsize=(12.8, 4.8))
    plt.fill_between(t, Qt_percentile_12_5, Qt_percentile_87_5, color='b', alpha=0.3, edgecolor='none')
    plt.fill_between(t, Qt_percentile_37_5, Qt_percentile_62_5, color='b', alpha=1)
    plt.plot(t, Qt_percentile_50, color='r', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Averga Tension Force Per Line')
    plt.title('Time evolution of Averga Tension Force Per Line (N)')
    plt.grid(True)
    plt.xlim(end_time - 30, end_time)
    plt.savefig('./results_figure/Tension_force_30s.png')
    
    
plot_quantiles()


    
