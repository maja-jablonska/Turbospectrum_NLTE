import os

def read_spectrum(filename):
    """
    Reads a Turbospectrum output file (Flux or Intensity).
    
    Parameters:
    -----------
    filename : str
        Path to the spectrum file.
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'wavelength': list of floats
        - 'flux_norm': list of floats
        - 'flux_abs': list of floats
        - 'mode': str ('Flux' or 'Intensity')
        - 'mu_points': list of floats (only if mode='Intensity')
        - 'intensity_abs': dict {mu: list of floats} (only if mode='Intensity')
        - 'intensity_norm': dict {mu: list of floats} (only if mode='Intensity')
    """
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            header_line = line
        elif line.strip():
            data_start = i
            break
        
    # Check for Intensity mode
    is_intensity = False
    mu_points = []
    
    if header_line and "mu-points" in header_line:
        is_intensity = True
        # Parse mu points
        # Header format: # mu-points   1.001800E-02  5.203500E-02 ...
        parts = header_line.split()
        if "mu-points" in parts:
            idx = parts.index("mu-points")
            mu_points = [float(x) for x in parts[idx+1:]]
            
    # Initialize lists
    wavelength = []
    flux_norm = []
    flux_abs = []
    
    i_abs_lists = {mu: [] for mu in mu_points}
    i_norm_lists = {mu: [] for mu in mu_points}
    
    for line in lines[data_start:]:
        parts = line.split()
        if not parts:
            continue
            
        try:
            # Common columns
            wavelength.append(float(parts[0]))
            flux_norm.append(float(parts[1]))
            flux_abs.append(float(parts[2]))
            
            if is_intensity:
                for i, mu in enumerate(mu_points):
                    # Columns are 0-indexed.
                    # i=0 -> cols 3, 4
                    col_abs = 3 + 2 * i
                    col_norm = 4 + 2 * i
                    
                    if col_abs < len(parts):
                        i_abs_lists[mu].append(float(parts[col_abs]))
                    if col_norm < len(parts):
                        i_norm_lists[mu].append(float(parts[col_norm]))
                        
        except (ValueError, IndexError):
            continue

    # Try to convert to numpy arrays if available
    try:
        import numpy as np
        wavelength = np.array(wavelength)
        flux_norm = np.array(flux_norm)
        flux_abs = np.array(flux_abs)
        if is_intensity:
            mu_points = np.array(mu_points)
            for mu in i_abs_lists:
                i_abs_lists[mu] = np.array(i_abs_lists[mu])
                i_norm_lists[mu] = np.array(i_norm_lists[mu])
    except ImportError:
        pass
    except ValueError: 
        # Catch the specific numpy error if it happens during import or usage
        pass

    result = {
        'mode': 'Intensity' if is_intensity else 'Flux',
        'wavelength': wavelength,
        'flux_norm': flux_norm,
        'flux_abs': flux_abs
    }
    
    if is_intensity:
        result['mu_points'] = mu_points
        result['intensity_abs'] = i_abs_lists
        result['intensity_norm'] = i_norm_lists
        
    return result

def load_all_spectra(directory):
    """
    Loads all .spec files in a directory.
    Returns a dict mapping filename to the read result.
    """
    results = {}
    for f in os.listdir(directory):
        if f.endswith(".spec"):
            path = os.path.join(directory, f)
            results[f] = read_spectrum(path)
    return results
