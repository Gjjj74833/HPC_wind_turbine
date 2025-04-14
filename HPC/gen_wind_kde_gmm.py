# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:29:26 2025

@author: ghhh7
"""

import joblib
import numpy as np
import sys
import os

# Example usage
# python gen_wind_kde_gmm.py {n_samples}, {method: gmm or kde}, {iteration index}
# generate wind signals use kde or gmm, with last element of each row as the seed to generate wave

# Load models
#pca = joblib.load('model_artifacts/pca_ite_0.pkl')

# Settings
num_samples = 1000
eps = 1e-12
T = 1502
dt = 1.0
N = T // 2 

n_array = 50
n = int(sys.argv[1])
method = sys.argv[2].lower()
ite = sys.argv[3]

samples = np.zeros((n, 750))

if  method == "kde":
    kde = joblib.load('model_artifacts/kde_MCMC.pkl')
    samples = kde.resample(n).T
    
elif method == "gmm":
    gmm = joblib.load('model_artifacts/gmm_MCMC.pkl')
    samples = gmm.sample(n)[0] 
    
#samples_log = pca.inverse_transform(samples)
samples = 10**(samples + eps)

original_phase = np.load('model_artifacts/original_phase.npy')

samples_complex = samples * np.exp(1j * original_phase)
sample_wind_speed = np.fft.irfft(samples_complex, n=T, axis=1)

rows_per_array = n // n_array

dirs = f'wind_{method}_ite_{ite}'

os.makedirs(dirs, exist_ok=True)

unique_ids = np.random.choice(np.arange(0, 9600000), size=n, replace=False)

for i in range(n_array):
    start_idx = i * rows_per_array
    end_idx = start_idx + rows_per_array
    sub_array = sample_wind_speed[start_idx:end_idx]
    sub_array = np.clip(sub_array, 3.0, 35.0)
    
    sub_ids = unique_ids[start_idx:end_idx].reshape(-1, 1)
    sub_array_with_ids = np.hstack([sub_array, sub_ids])
    
    filename = f"{dirs}/wind_{i+1}.npy"
    np.save(filename, sub_array_with_ids)
    print(f"Saved: {filename}")
