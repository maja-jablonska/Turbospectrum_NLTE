import sys
import os

# Add the parent directory to the Python path to be able to import the wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbospectrum_wrapper.wrapper import Turbospectrum, ModelAtmosphere, Linelist, Abundances

def main():
    # Set the path to the Turbospectrum root directory
    # IMPORTANT: You need to modify this path to match your local installation
    ts_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Create a Turbospectrum instance
    ts = Turbospectrum(ts_root=ts_root)

    # Define the grid of parameters
    # For simplicity, we will vary the metallicity of Fe
    param_grid = [
        {
            'lambda_min': 6400.0, 'lambda_max': 6800.0, 'lambda_step': 0.01,
            'model_atmosphere': ModelAtmosphere(path='COM/contopac/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.modopac'),
            'abundances': Abundances(individual_abundances={26: 7.40}),
            'linelists': [Linelist(path='linelists/nlte_linelist_test.txt'), Linelist(path='DATA/Hlinedata')],
            'result_file': 'syntspec/grid_spectrum_1.spec'
        },
        {
            'lambda_min': 6400.0, 'lambda_max': 6800.0, 'lambda_step': 0.01,
            'model_atmosphere': ModelAtmosphere(path='COM/contopac/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.modopac'),
            'abundances': Abundances(individual_abundances={26: 7.50}),
            'linelists': [Linelist(path='linelists/nlte_linelist_test.txt'), Linelist(path='DATA/Hlinedata')],
            'result_file': 'syntspec/grid_spectrum_2.spec'
        },
        {
            'lambda_min': 6400.0, 'lambda_max': 6800.0, 'lambda_step': 0.01,
            'model_atmosphere': ModelAtmosphere(path='COM/contopac/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.modopac'),
            'abundances': Abundances(individual_abundances={26: 7.60}),
            'linelists': [Linelist(path='linelists/nlte_linelist_test.txt'), Linelist(path='DATA/Hlinedata')],
            'result_file': 'syntspec/grid_spectrum_3.spec'
        },
    ]

    # Run the grid of calculations in parallel
    spectra = ts.run_grid(param_grid)

    # Print the results
    for i, spectrum in enumerate(spectra):
        print(f"--- Spectrum {i+1} ---")
        print(f"Fe abundance: {param_grid[i]['abundances'].individual_abundances[26]}")
        # print the first 5 points of the spectrum
        for j in range(5):
            print(f"{spectrum.wavelength[j]:.4f}  {spectrum.flux[j]:.4f}")
        print("...")

if __name__ == "__main__":
    main()
