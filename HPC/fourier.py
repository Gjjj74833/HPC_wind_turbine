# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:11:44 2023

@author: Yihan Liu

perform fourier transform analysis frequency 
"""
import os
import numpy as np
import matplotlib.pyplot as plt


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
    seeds = [data['seeds'] for data in datas]
    
    # Concatenate all the collected data (only one concatenation operation per field)
    state = np.concatenate(states, axis=2)
    wind_speed = np.hstack(wind_speeds)
    wave_eta = np.hstack(wave_etas)
    Q_t = np.hstack(Q_ts)
    seeds = np.hstack(seeds)
    
    return t, state, wind_speed, wave_eta, seeds, Q_t


def fourier_transform(state, time_interval, make_plot = False, state_name = '', destination_directory = ''):
    """
    given an 1D array and desired time_interval to perform fourier transform
    make plot

    Parameters
    ----------
    state : 1 dimensional array
        the original signal to transform
    time_interval : 1 dimensional array
        the time interval to operate.

    Returns
    -------
    the transfered frequency function

    """
    
    sample_rate = np.average(np.diff(time_interval)) 
    

    # Perform the Fourier Transform on your 'wave_eta' data
    yf = np.fft.fft(state)
    tf = np.fft.fftfreq(n=len(state), d=sample_rate)

    # As the fft results are symmetrical, we only need to take the first half
    # Also, it's common to take the absolute value because the FFT returns complex numbers.
    tf = tf[:len(tf)//2]
    yf = yf[:len(yf)//2]

    # Create a plot of the positive frequencies against the magnitudes
    if make_plot:
        plt.figure()
        plt.plot(tf, np.abs(yf))  
        plt.title(f'Frequency Domain of {state_name}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.savefig(f'{destination_directory}/fre_{state_name}.png', dpi=300)






