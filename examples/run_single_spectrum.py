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

if __name__ == "__main__":
    main()
