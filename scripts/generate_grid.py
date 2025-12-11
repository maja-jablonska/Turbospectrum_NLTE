import csv
import math
from itertools import islice, product

import numpy as np
import yaml


DEFAULT_PARAMETER_CONFIG = {
    'teff': {'min': 5000, 'max': 6250, 'step': 250},
    'logg': {'min': 3.5, 'max': 5.0, 'step': 0.5},
    'feh': {'min': -1.0, 'max': 0.5, 'step': 0.5},
    't_value': ["01"],
    'a': ["+0.00"],
    'c': ["+0.00"],
    'n': ["+0.00"],
    'o': ["+0.00"],
    'r': ["+0.00"],
    's': ["+0.00"],
}


def _format_abundance_value(value):
    if isinstance(value, (int, float, np.floating)):
        return f"{value:+.2f}"
    return value


def _resolve_values(name, cfg, rng, is_abundance=False, fallback_count=None):
    """Return a list of values for a parameter from lists or distributions."""
    if cfg is None:
        cfg = DEFAULT_PARAMETER_CONFIG.get(name)

    if isinstance(cfg, (str, float, int, np.floating)):
        values = [cfg]
    elif isinstance(cfg, list):
        values = cfg
    elif isinstance(cfg, dict):
        if 'values' in cfg:
            values = cfg['values'] if isinstance(cfg['values'], list) else [cfg['values']]
        elif {'min', 'max', 'step'}.issubset(cfg):
            values = np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step'])
        elif 'distribution' in cfg:
            dist = cfg['distribution'] or {}
            dist_type = dist.get('type', 'gaussian')
            count = dist.get('count', fallback_count)
            if count is None:
                raise ValueError(f"Distribution for '{name}' requires a 'count' or sampling max_rows")
            if rng is None:
                rng = np.random.default_rng()
            if dist_type == 'gaussian':
                mean = dist.get('mean', 0.0)
                sigma = dist.get('sigma', 0.1)
                values = rng.normal(mean, sigma, int(count))
            elif dist_type == 'uniform':
                min_val = dist.get('min')
                max_val = dist.get('max')
                if min_val is None or max_val is None:
                    raise ValueError(f"Uniform distribution for '{name}' requires 'min' and 'max'")
                values = rng.uniform(min_val, max_val, int(count))
            else:
                raise ValueError(f"Unsupported distribution type '{dist_type}' for parameter '{name}'")
        else:
            raise ValueError(f"Could not parse configuration for parameter '{name}'")
    else:
        raise ValueError(f"Unsupported configuration type for parameter '{name}'")

    if is_abundance:
        return [_format_abundance_value(v) for v in values]
    return list(values)


def _sample_parameter_space(parameter_lists, sampling_cfg, rng):
    method = (sampling_cfg or {}).get('method', 'full')
    max_rows = (sampling_cfg or {}).get('max_rows')

    values_only = list(parameter_lists.values())

    if method == 'full':
        combos = product(*values_only)
        return islice(combos, max_rows) if max_rows else combos

    if max_rows is None:
        raise ValueError(f"Sampling method '{method}' requires 'max_rows' to be set")

    if rng is None:
        rng = np.random.default_rng()

    if method == 'random':
        return (tuple(rng.choice(values) for values in values_only) for _ in range(int(max_rows)))

    if method == 'latin_hypercube':
        permutations = []
        for values in values_only:
            repeats = math.ceil(max_rows / len(values))
            base_indices = np.tile(np.arange(len(values)), repeats)[: int(max_rows)]
            permutations.append(np.asarray(base_indices)[rng.permutation(int(max_rows))])

        def lhs_generator():
            for row_idx in range(int(max_rows)):
                yield tuple(values[idx_list[row_idx]] for values, idx_list in zip(values_only, permutations))

        return lhs_generator()

    raise ValueError(f"Unsupported sampling method '{method}'")


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

    sampling_cfg = config.get('sampling', {})
    rng = np.random.default_rng(sampling_cfg.get('seed')) if sampling_cfg.get('seed') is not None else None
    fallback_count = sampling_cfg.get('max_rows')

    atmosphere_cfg = config.get('atmosphere', {})
    abundance_cfg = config.get('abundance', {})

    param_names = ['teff', 'logg', 'feh', 't_value', 'a', 'c', 'n', 'o', 'r', 's']
    parameter_lists = {}
    for name in param_names:
        section = atmosphere_cfg if name in ['teff', 'logg', 'feh', 't_value'] else abundance_cfg
        parameter_lists[name] = _resolve_values(name, section.get(name), rng, is_abundance=name in abundance_cfg or name in ['a', 'c', 'n', 'o', 'r', 's'], fallback_count=fallback_count)

    synthesis_cfg = config.get('synthesis', config)
    lam_min = synthesis_cfg.get('lam_min', 6000)
    lam_max = synthesis_cfg.get('lam_max', 6100)
    lam_step = synthesis_cfg.get('lam_step', 0.01)
    turbvel = config.get('turbvel', "01")
    output_mode = config.get('output_mode', 'Flux')
    mode = config.get('mode', '1D')
    calculation_mode = config.get('calculation_mode', 'LTE')
    grid_version = config.get('grid_version', 'unversioned')

    parameter_combinations = _sample_parameter_space(parameter_lists, sampling_cfg, rng)

    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'grid_version', 'teff', 'logg', 'feh', 'lam_min', 'lam_max', 'lam_step',
                'turbvel', 't_value', 'a', 'c', 'n', 'o', 'r', 's', 'output_mode', 'mode', 'calculation_mode'
            ])

            for teff, logg, feh, t_val, a_val, c_val, n_val, o_val, r_val, s_val in parameter_combinations:
                writer.writerow([
                    grid_version,
                    int(teff),
                    round(float(logg), 2),
                    round(float(feh), 2),
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
