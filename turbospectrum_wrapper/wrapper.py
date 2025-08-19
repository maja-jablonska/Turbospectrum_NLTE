from typing import Optional, Dict, List, Any

class ModelAtmosphere:
    """
    Represents a model atmosphere file.

    Args:
        path (str): The path to the model atmosphere file.
    """
    def __init__(self, path: str):
        self.path = path

class Linelist:
    """
    Represents a line list file.

    Args:
        path (str): The path to the line list file.
    """
    def __init__(self, path: str):
        self.path = path

class Abundances:
    """
    Represents the chemical abundances.

    Args:
        metallicity (float, optional): The overall metallicity [Fe/H]. Defaults to 0.0.
        alpha_fe (float, optional): The alpha-to-iron ratio [alpha/Fe]. Defaults to 0.0.
        he_h (float, optional): The helium-to-hydrogen ratio [He/H]. Defaults to 0.0.
        r_process (float, optional): The r-process element abundance scaling. Defaults to 0.0.
        s_process (float, optional): The s-process element abundance scaling. Defaults to 0.0.
        individual_abundances (Optional[Dict[int, float]], optional): A dictionary of individual element abundances, where the key is the atomic number and the value is the abundance. Defaults to None.
        isotopes (Optional[Dict[float, float]], optional): A dictionary of isotopic fractions, where the key is the isotope mass number and the value is the fraction. Defaults to None.
        abund_source (str, optional): The source of the solar abundances. Can be 'magg', 'asp2007', or 'gs1998'. Defaults to 'magg'.
    """
    def __init__(self, metallicity: float = 0.0, alpha_fe: float = 0.0, he_h: float = 0.0, r_process: float = 0.0, s_process: float = 0.0, individual_abundances: Optional[Dict[int, float]] = None, isotopes: Optional[Dict[float, float]] = None, abund_source: str = 'magg'):
        self.metallicity = metallicity
        self.alpha_fe = alpha_fe
        self.he_h = he_h
        self.r_process = r_process
        self.s_process = s_process
        self.individual_abundances = individual_abundances or {}
        self.isotopes = isotopes or {}
        self.abund_source = abund_source

class Spectrum:
    """
    Represents the output spectrum.

    Args:
        wavelength (List[float]): The wavelength points of the spectrum.
        flux (List[float]): The flux values of the spectrum.
        flux_normalized (Optional[List[float]], optional): The continuum-normalized flux. Defaults to None.
        intensity_mu1 (Optional[List[float]], optional): The intensity at mu=1. Defaults to None.
        continuum (Optional[List[float]], optional): The continuum flux. Defaults to None.
        intensity (Optional[Dict[float, List[float]]], optional): A dictionary of intensity spectra at different angles, where the key is the mu value. Defaults to None.
    """
    def __init__(self, wavelength: List[float], flux: List[float], flux_normalized: Optional[List[float]] = None, intensity_mu1: Optional[List[float]] = None, continuum: Optional[List[float]] = None, intensity: Optional[Dict[float, List[float]]] = None):
        self.wavelength = wavelength
        self.flux = flux
        self.flux_normalized = flux_normalized
        self.intensity_mu1 = intensity_mu1
        self.continuum = continuum
        self.intensity = intensity

class Turbospectrum:
    """
    The main class to control the execution of Turbospectrum.

    Args:
        ts_root (str): The path to the Turbospectrum root directory.
    """
    def __init__(self, ts_root: str):
        self.ts_root = ts_root
        # Paths to the Turbospectrum executables
        self.babsma_path = f"{ts_root}/exec-gf/babsma_lu"
        self.bsyn_path = f"{ts_root}/exec-gf/bsyn_lu"

    def configure(self, lambda_min: float, lambda_max: float, lambda_step: float, model_atmosphere: ModelAtmosphere, abundances: Abundances, linelists: List[Linelist], spherical: bool = False, intensity_flux: str = 'Flux', nlte: bool = False, nlte_info_file: Optional[str] = None, result_file: str = 'syntspec/spectrum.spec'):
        """
        Configures a Turbospectrum run by generating the input script for the bsyn executable.

        Args:
            lambda_min (float): The minimum wavelength of the spectral synthesis.
            lambda_max (float): The maximum wavelength of the spectral synthesis.
            lambda_step (float): The wavelength step of the spectral synthesis.
            model_atmosphere (ModelAtmosphere): The model atmosphere to use.
            abundances (Abundances): The chemical abundances to use.
            linelists (List[Linelist]): A list of line lists to use.
            spherical (bool, optional): Whether to use spherical geometry. Defaults to False.
            intensity_flux (str, optional): Whether to compute intensity or flux. Can be 'Intensity' or 'Flux'. Defaults to 'Flux'.
            nlte (bool, optional): Whether to perform NLTE calculations. Defaults to False.
            nlte_info_file (Optional[str], optional): The path to the NLTE info file. Defaults to None.
            result_file (str, optional): The path to the output spectrum file. Defaults to 'syntspec/spectrum.spec'.
        """
        self.result_file = result_file
        self.bsyn_input = f"""
'NLTE :'          '{".true." if nlte else ".false."}'
'NLTEINFOFILE:'  '{nlte_info_file or "DATA/SPECIES_LTE_NLTE.dat"}'
'LAMBDA_MIN:'     '{lambda_min}'
'LAMBDA_MAX:'     '{lambda_max}'
'LAMBDA_STEP:'    '{lambda_step}'
'INTENSITY/FLUX:' '{intensity_flux}'
'MODELOPAC:' '{model_atmosphere.path}'
'RESULTFILE :' '{result_file}'
'ABUND_SOURCE:'   '{abundances.abund_source}'
'METALLICITY:'    '{abundances.metallicity}'
'ALPHA/Fe   :'    '{abundances.alpha_fe}'
'HELIUM     :'    '{abundances.he_h}'
'R-PROCESS  :'    '{abundances.r_process}'
'S-PROCESS  :'    '{abundances.s_process}'
'INDIVIDUAL ABUNDANCES:'   '{len(abundances.individual_abundances)}'
"""
        for element, abundance in abundances.individual_abundances.items():
            self.bsyn_input += f"{element}  {abundance}\n"
        self.bsyn_input += f"'ISOTOPES : ' '{len(abundances.isotopes)}'\n"
        for isotope, fraction in abundances.isotopes.items():
            self.bsyn_input += f"{isotope}  {fraction}\n"
        self.bsyn_input += f"'NFILES   :' '{len(linelists)}'\n"
        for linelist in linelists:
            self.bsyn_input += f"{linelist.path}\n"
        self.bsyn_input += f"'SPHERICAL:'  '{'T' if spherical else 'F'}'\n"
        self.bsyn_input += "  30\n  300.00\n  15\n  1.30\n"

    def run_bsyn(self, intensity_flux: str = 'Flux') -> Spectrum:
        """
        Runs the bsyn executable and parses the output spectrum file.

        Args:
            intensity_flux (str, optional): Whether to parse an intensity or flux spectrum. Defaults to 'Flux'.

        Returns:
            Spectrum: The resulting spectrum.
        """
        import subprocess
        import numpy as np
        import re

        process = subprocess.run(
            [self.bsyn_path],
            input=self.bsyn_input,
            text=True,
            capture_output=True,
            check=True
        )

        with open(self.result_file, 'r') as f:
            header = ""
            for line in f:
                if line.startswith('#'):
                    header += line
                else:
                    break

        data = np.loadtxt(self.result_file)
        wavelength = data[:, 0]

        if intensity_flux == 'Intensity':
            # In intensity mode, Turbospectrum can output many columns.
            # The first is wavelength, the second is normalized flux, the third is flux.
            # The following columns are intensities at different mu angles.
            # The file can also have a header with mu values.
            flux_normalized = data[:, 1]
            flux = data[:, 2]

            # Check for mu values in the header
            mu_values_match = re.search(r'#\s*mu\s*=\s*(.*)', header)
            if mu_values_match:
                mu_values_str = mu_values_match.group(1).split()
                mu_values = [float(mu) for mu in mu_values_str]
            else:
                # Fallback to default mu values if not in header
                mu_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

            intensities = {}
            for i in range(3, data.shape[1]):
                mu_index = i - 3
                if mu_index < len(mu_values):
                    mu = mu_values[mu_index]
                    intensities[mu] = data[:, i]

            intensity_mu1 = intensities.get(1.0)

            return Spectrum(wavelength=wavelength, flux=flux, flux_normalized=flux_normalized, intensity_mu1=intensity_mu1, intensity=intensities)
        else:
            # flux mode
            flux = data[:, 1]
            continuum = None
            if data.shape[1] > 2:
                continuum = data[:, 2]
            return Spectrum(wavelength=wavelength, flux=flux, continuum=continuum)

    def run_grid(self, params: List[Dict[str, Any]]) -> List[Spectrum]:
        """
        Runs a grid of Turbospectrum calculations in parallel.

        Args:
            params (List[Dict[str, Any]]): A list of parameter dictionaries. Each dictionary should contain the arguments for the `configure` method.

        Returns:
            List[Spectrum]: A list of the resulting spectra.
        """
        import multiprocessing

        with multiprocessing.Pool() as pool:
            results = pool.map(self._run_single_calculation, params)

        return results

    def _run_single_calculation(self, param_dict: Dict[str, Any]) -> Spectrum:
        """
        Helper function to run a single Turbospectrum calculation. This function is used by `run_grid`.

        Args:
            param_dict (Dict[str, Any]): A dictionary of parameters for the calculation.

        Returns:
            Spectrum: The resulting spectrum.
        """
        intensity_flux = param_dict.get('intensity_flux', 'Flux')
        self.configure(**param_dict)
        return self.run_bsyn(intensity_flux=intensity_flux)
