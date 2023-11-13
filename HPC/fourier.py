# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:11:44 2023

@author: Yihan Liu

perform fourier transform analysis frequency 
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_fft(signal, sample_rate):
    # Perform the FFT
    fft_result = np.fft.fft(signal)
    
    # Calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fft_result)
    
    # Create the corresponding frequency values for the x-axis
    frequencies = np.fft.fftfreq(len(signal), 1.0 / sample_rate)
    
    # Plot the amplitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, amplitude_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Spectrum (FFT)')
    plt.grid(True)
    plt.show()

# Example usage:
# Generate some example data
sample_rate = 1000  # Sample rate in Hz
t = np.linspace(0, 1, sample_rate, endpoint=False)  # Time values from 0 to 1
frequency = 5  # Frequency of the sinusoid
amplitude = 1.0  # Amplitude of the sinusoid
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Call the function with the signal data and sample rate
plot_fft(signal, sample_rate)