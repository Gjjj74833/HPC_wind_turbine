# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:43:36 2024

@author: ghhh7
"""

import os
import numpy as np
from scipy.signal import welch
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Computes the Power Spectrum Density (PSD) of a time series with constant time steps.
    
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
    wind_10_20 = []
    wind_20 = []
    
    # Iterate over each simulation index
    for i in range(wind_speed.shape[1]):
        avg_wind_speed = np.mean(wind_speed[:, i])
        
        if avg_wind_speed < 10:
            wind_0_10.append(i)
        elif 10 <= avg_wind_speed < 20:
            wind_10_20.append(i)
        else:
            wind_20.append(i)
    
    return wind_0_10, wind_10_20, wind_20

def save_psd_percentiles(psd_list, file_name):
    """
    Calculates and saves the median, 12.5th percentile, 87.5th percentile,
    37.5th percentile, and 62.5th percentile of a list of PSDs.
    
    Parameters:
    psd_list (list of numpy arrays): List of PSD arrays.
    file_name (str): The name of the file to save the percentiles.
    """
    psd_array = np.array(psd_list)
    median = np.median(psd_array, axis=0)
    p12_5 = np.percentile(psd_array, 12.5, axis=0)
    p87_5 = np.percentile(psd_array, 87.5, axis=0)
    p37_5 = np.percentile(psd_array, 37.5, axis=0)
    p62_5 = np.percentile(psd_array, 62.5, axis=0)
    
    np.savez(file_name, median=median, p12_5=p12_5, p87_5=p87_5, p37_5=p37_5, p62_5=p62_5)

def compute_psd_for_category(indices, wind_speed, wave_eta, state, time_step):
    psd_wind = []
    psd_wave = []
    psd_surge = []
    
    for i in indices:
        psd_wind.append(psd(wind_speed[:, i], time_step))
        psd_wave.append(psd(wave_eta[:, i], time_step))
        psd_surge.append(psd(state[:, 0, i], time_step))
    
    return psd_wind, psd_wave, psd_surge

# Load data
t, state, wind_speed, wave_eta, seeds = load_data("results")

# Categorize simulations
wind_0_10, wind_10_20, wind_20 = categorize_simulations_by_wind_speed(wind_speed)

# Initialize lists to store PSDs
psd_wind_0_10 = []
psd_wind_10_20 = []
psd_wind_20 = []

psd_wave_0_10 = []
psd_wave_10_20 = []
psd_wave_20 = []

psd_surge_0_10 = []
psd_surge_10_20 = []
psd_surge_20 = []

time_step = t[1] - t[0]

# Use ProcessPoolExecutor to parallelize PSD computations
with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(compute_psd_for_category, indices, wind_speed, wave_eta, state, time_step): category
        for indices, category in zip([wind_0_10, wind_10_20, wind_20], ['wind_0_10', 'wind_10_20', 'wind_20'])
    }
    
    for future in as_completed(futures):
        category = futures[future]
        try:
            psd_wind, psd_wave, psd_surge = future.result()
            if category == 'wind_0_10':
                psd_wind_0_10.extend(psd_wind)
                psd_wave_0_10.extend(psd_wave)
                psd_surge_0_10.extend(psd_surge)
            elif category == 'wind_10_20':
                psd_wind_10_20.extend(psd_wind)
                psd_wave_10_20.extend(psd_wave)
                psd_surge_10_20.extend(psd_surge)
            elif category == 'wind_20':
                psd_wind_20.extend(psd_wind)
                psd_wave_20.extend(psd_wave)
                psd_surge_20.extend(psd_surge)
        except Exception as exc:
            print(f"Generated an exception: {exc}")

# Save percentiles and save to file
save_psd_percentiles(psd_wind_0_10, "psd_wind_0_10.npz")
save_psd_percentiles(psd_wind_10_20, "psd_wind_10_20.npz")
save_psd_percentiles(psd_wind_20, "psd_wind_20.npz")

save_psd_percentiles(psd_wave_0_10, "psd_wave_0_10.npz")
save_psd_percentiles(psd_wave_10_20, "psd_wave_10_20.npz")
save_psd_percentiles(psd_wave_20, "psd_wave_20.npz")

save_psd_percentiles(psd_surge_0_10, "psd_surge_0_10.npz")
save_psd_percentiles(psd_surge_10_20, "psd_surge_10_20.npz")
save_psd_percentiles(psd_surge_20, "psd_surge_20.npz")