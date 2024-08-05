# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:05:37 2024

@author: ghhh7
"""
import os
import numpy as np
from scipy.signal import welch

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
    
    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    wind_speed = np.hstack(wind_speeds)
    wave_eta = np.hstack(wave_etas)
    seeds = np.hstack(seeds)
    
    return t, state, wind_speed, wave_eta, seeds

def psd(time_series, time_step = 0.5, nperseg=256):
    """
    Plots the Power Spectrum Density (PSD) of a time series with constant time steps.
    
    Parameters:
    time_series (numpy array): The time series data.
    time_step (float): The time step between consecutive samples.
    nperseg (int): Length of each segment for Welch's method. Default is 256.
    """
    # Calculate the sampling frequency
    fs = 1 / time_step
    
    # Compute the Power Spectrum Density using Welch's method
    frequencies, psd = welch(time_series, fs, nperseg=nperseg)
    
    return psd


def categorize_simulations_by_wind_speed(wind_speed):
    wind_0_10 = []
    wind_20 = []
    
    # Iterate over each simulation index
    for i in range(wind_speed.shape[1]):
        avg_wind_speed = np.mean(wind_speed[:, i])
        
        if avg_wind_speed < 12:
            wind_0_10.append(i)
        else:
            wind_20.append(i)
    
    return wind_0_10, wind_20

def save_psd_percentiles(psd_list, file_name):
    """
    Calculates and saves the median, 25th percentile, and 75th percentile of a list of PSDs.
    
    Parameters:
    psd_list (list of numpy arrays): List of PSD arrays.
    
    Returns:
    tuple: median, 25th percentile, and 75th percentile
    """
    psd_array = np.array(psd_list)
    median = np.median(psd_array, axis=0)
    p12_5 = np.percentile(psd_array, 12.5, axis=0)
    p87_5 = np.percentile(psd_array, 87.5, axis=0)
    p37_5 = np.percentile(psd_array, 37.5, axis=0)
    p62_5 = np.percentile(psd_array, 62.5, axis=0)
    
    np.savez(file_name, median=median, p12_5=p12_5, p87_5=p87_5,
                                       p37_5=p37_5, p62_5=p62_5)

# Load data
t, state, wind_speed, wave_eta, seeds = load_data("results")

# Categorize simulations
wind_0_10, wind_20 = categorize_simulations_by_wind_speed(wind_speed)

psd_wind_0_10 = []
psd_wind_20 = []

psd_wave_0_10 = []
psd_wave_20 = []

psd_surge_0_10 = []
psd_surge_20 = []              

for i in wind_0_10:
    psd_wind_0_10.append(psd(wind_speed[:, i]))
    psd_wave_0_10.append(psd(wave_eta[:, i]))
    psd_surge_0_10.append(psd(state[:, 0, i]))

    

for i in wind_20:
    psd_wind_20.append(psd(wind_speed[:, i]))
    psd_wave_20.append(psd(wave_eta[:, i]))
    psd_surge_20.append(psd(state[:, 0, i]))


save_psd_percentiles(psd_wind_0_10, "psd/wind_0_10")
save_psd_percentiles(psd_wind_20, "psd/wind_20")

save_psd_percentiles(psd_wave_0_10, "psd/wave_0_10")
save_psd_percentiles(psd_wave_20, "psd/wave_20")

save_psd_percentiles(psd_surge_0_10, "psd/surge_0_10")
save_psd_percentiles(psd_surge_20, "psd/surge_20")


