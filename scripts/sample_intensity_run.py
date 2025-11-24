import os
import sys

# Add the scripts directory to the path so we can import run_turbospectrum
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from run_turbospectrum import TurbospectrumConfig, run_grid

def main():
    # Detect project root
    project_root = os.path.dirname(script_dir)
    
    # Initialize configuration
    config = TurbospectrumConfig(project_root=project_root)
    
    # Enable Intensity Calculation
    config.calculate_intensity = True
    
    # Note: Turbospectrum now automatically calculates intensities for standard mu angles
    # when Intensity mode is enabled, so we don't need to specify them explicitly here.
    # config.mu_angles = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1] # Deprecated
    
    # Define Grid Points
    # Format: (Teff, logg, Fe/H, microturb_str)
    # Using available models from the directory + one for interpolation
    grid_points = [
        (2500, 3.0, 0.00, "01"), 
        (2550, 3.0, 0.00, "01"), # Should trigger interpolation between 2500 and 2600
    ]
    
    print("Starting sample intensity run...")
    print(f"Calculating intensity for all standard mu angles.")
    
    # Run the grid (Intensity)
    run_grid(config, grid_points)
    
    # Run for Flux as well for comparison
    print("\nStarting sample flux run...")
    config.calculate_intensity = False
    run_grid(config, grid_points)
    
    # Verify Reading
    print("\n--- Verifying Output Reading ---")
    from turbospectrum_utils import read_spectrum
    
    output_dir = config.output_dir
    
    # Check one intensity file
    int_file = os.path.join(output_dir, "p2550_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.intensity.spec")
    if os.path.exists(int_file):
        print(f"Reading {os.path.basename(int_file)}...")
        data = read_spectrum(int_file)
        print(f"Mode: {data['mode']}")
        print(f"Wavelength points: {len(data['wavelength'])}")
        if data['mode'] == 'Intensity':
            print(f"Mu points: {data['mu_points']}")
            # Handle list or numpy array
            shape_info = "List length: " + str(len(data['intensity_abs'][data['mu_points'][0]]))
            try:
                shape_info = "Array shape: " + str(data['intensity_abs'][data['mu_points'][0]].shape)
            except:
                pass
            print(f"Intensity Abs Info: {shape_info}")
            
            val = data['intensity_abs'][data['mu_points'][0]][0]
            print(f"Sample Intensity (mu={data['mu_points'][0]}): {val}")
            
    # Check one flux file
    flux_file = os.path.join(output_dir, "p2550_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.spec")
    if os.path.exists(flux_file):
        print(f"\nReading {os.path.basename(flux_file)}...")
        data = read_spectrum(flux_file)
        print(f"Mode: {data['mode']}")
        print(f"Flux Abs Points: {len(data['flux_abs'])}")
        print(f"Sample Flux: {data['flux_abs'][0]}")

if __name__ == "__main__":
    main()
