import yaml
import csv
import numpy as np
from itertools import product

def generate_grid(config_path='scripts/grid_config.yml', output_path='scripts/parameter_grid.csv'):
    """
    Generates a CSV parameter grid based on a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        output_path (str): Path to the output CSV file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Generate the grid points for each stellar parameter
    teff_range = np.arange(config['teff']['min'], config['teff']['max'] + config['teff']['step'], config['teff']['step'])
    logg_range = np.arange(config['logg']['min'], config['logg']['max'] + config['logg']['step'], config['logg']['step'])
    feh_range = np.arange(config['feh']['min'], config['feh']['max'] + config['feh']['step'], config['feh']['step'])

    # Handle t_value as a list or single value
    t_value_list = config.get('t_value', ["01"])
    if not isinstance(t_value_list, list):
        t_value_list = [t_value_list]

    # Handle abundance parameters as lists or single values
    a_list = config.get('a', ["+0.00"])
    if not isinstance(a_list, list):
        a_list = [a_list]
    c_list = config.get('c', ["+0.00"])
    if not isinstance(c_list, list):
        c_list = [c_list]
    n_list = config.get('n', ["+0.00"])
    if not isinstance(n_list, list):
        n_list = [n_list]
    o_list = config.get('o', ["+0.00"])
    if not isinstance(o_list, list):
        o_list = [o_list]
    r_list = config.get('r', ["+0.00"])
    if not isinstance(r_list, list):
        r_list = [r_list]
    s_list = config.get('s', ["+0.00"])
    if not isinstance(s_list, list):
        s_list = [s_list]

    # Get the synthesis and calculation parameters
    lam_min = config.get('lam_min', 6000)
    lam_max = config.get('lam_max', 6100)
    lam_step = config.get('lam_step', 0.01)
    turbvel = config.get('turbvel', "01")
    output_mode = config.get('output_mode', 'Flux')
    mode = config.get('mode', '1D')
    calculation_mode = config.get('calculation_mode', 'LTE')

    # Create the Cartesian product of all parameter ranges
    parameter_combinations = product(
        teff_range,
        logg_range,
        feh_range,
        t_value_list,
        a_list,
        c_list,
        n_list,
        o_list,
        r_list,
        s_list
    )

    # Write the grid to a CSV file
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Update header to include all new parameters
            writer.writerow([
                'teff', 'logg', 'feh', 'lam_min', 'lam_max', 'lam_step', 'turbvel',
                't_value', 'a', 'c', 'n', 'o', 'r', 's', 'output_mode', 'mode', 'calculation_mode'
            ])

            # Write the parameter combinations
            for teff, logg, feh, t_val, a_val, c_val, n_val, o_val, r_val, s_val in parameter_combinations:
                writer.writerow([
                    int(teff),
                    round(logg, 2),
                    round(feh, 2),
                    lam_min,
                    lam_max,
                    lam_step,
                    turbvel,
                    t_val,
                    a_val,
                    c_val,
                    n_val,
                    o_val,
                    r_val,
                    s_val,
                    output_mode,
                    mode,
                    calculation_mode
                ])
        print(f"Successfully generated parameter grid at {output_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_path}")

if __name__ == '__main__':
    generate_grid()