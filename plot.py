#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np  
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys
import os
# Add scripts directory to path to allow import
sys.path.append(os.path.abspath('scripts'))

from turbospectrum_utils import read_spectrum


# In[ ]:


# Define file path (using one of the generated intensity files)
file = 'spectra/p2550_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.intensity.spec'

if os.path.exists(file):
    # Read spectrum using the utility
    data = read_spectrum(file)
    
    wave = data['wavelength']
    flux_norm = data['flux_norm']
    
    plt.figure(figsize=(10, 6))
    plt.plot(wave, flux_norm, label='Normalized Flux')
    
    # If it's an intensity file, we can also plot specific mu angles
    if data['mode'] == 'Intensity':
        # Plot the first mu point intensity
        mu0 = data['mu_points'][0]
        # Handle list or numpy array
        try:
             i_norm = data['intensity_norm'][mu0]
        except:
             i_norm = data['intensity_norm'][mu0]
             
        plt.plot(wave, i_norm, label=f'Intensity (mu={mu0:.2f})', alpha=0.7)

    plt.xlabel('Wavelength (A)')
    plt.ylabel('Normalized Flux/Intensity')
    plt.title(f'Turbospectrum Output: {os.path.basename(file)}')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"File not found: {file}")

