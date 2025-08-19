# Turbospectrum Wrapper

A modern Python wrapper for the Turbospectrum stellar synthesis code.

## Overview

This wrapper provides a high-level, object-oriented interface to Turbospectrum, making it easy to run spectral synthesis calculations from Python. It is designed to be flexible and extensible, and it supports running large grids of calculations in parallel.

## Features

-   Object-oriented interface to Turbospectrum.
-   Support for running large grids of calculations in parallel.
-   Support for NLTE calculations.
-   Support for intensity and continuum output.
-   Easy to use and extend.

## Installation

1.  **Install Turbospectrum:**
    Before using this wrapper, you need to have a working installation of Turbospectrum. You can download it from the official GitHub repository: [https://github.com/bertrandplez/Turbospectrum2020](https://github.com/bertrandplez/Turbospectrum2020)

2.  **Clone this repository:**
    ```bash
    git clone <repository_url>
    ```

## Quick Start

Here is a simple example of how to use the wrapper to generate a synthetic spectrum:

```python
from wrapper import Turbospectrum, ModelAtmosphere, Linelist, Abundances

# Set the path to the Turbospectrum root directory
ts_root = '/path/to/your/Turbospectrum2020'

# Create a Turbospectrum instance
ts = Turbospectrum(ts_root=ts_root)

# Define the parameters for the spectral synthesis
model_atmosphere = ModelAtmosphere(path='COM/contopac/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.modopac')
abundances = Abundances(individual_abundances={26: 7.50})
linelists = [Linelist(path='linelists/nlte_linelist_test.txt'), Linelist(path='DATA/Hlinedata')]

# Configure the Turbospectrum run
ts.configure(
    lambda_min=6400.0,
    lambda_max=6800.0,
    lambda_step=0.01,
    model_atmosphere=model_atmosphere,
    abundances=abundances,
    linelists=linelists,
    result_file='syntspec/single_spectrum.spec'
)

# Run the spectral synthesis
spectrum = ts.run_bsyn()

# Print the resulting spectrum
print("Generated spectrum:")
for i in range(len(spectrum.wavelength)):
    print(f"{spectrum.wavelength[i]:.4f}  {spectrum.flux[i]:.4f}")
```

For more examples, see the `examples` directory.
