import os
import sys
import subprocess
import multiprocessing
import time
import glob
import re
import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TurbospectrumConfig:
    # Paths
    project_root: str
    compiler: str = "gf"  # 'gf' or 'intel'
    # NLTE options
    nlte: bool = False
    nlte_info_file: str = ""
    force: bool = False
    
    # Input Directories (Absolute paths recommended)
    model_atmosphere_path: str = ""
    linelist_path: str = ""
    linelist_files: List[str] = None
    
    # Output Directories
    output_dir: str = ""
    log_dir: str = ""
    tmp_dir: str = ""
    
    # Executable Names
    babsma_exec: str = "babsma_lu"
    bsyn_exec: str = "bsyn_lu"
    interpol_exec: str = "interpol_modeles"

    # Intensity Calculation
    calculate_intensity: bool = False
    mu_angles: List[float] = field(default_factory=list)

    # Synthesis Parameters
    lambda_min: float = 4000
    lambda_max: float = 8000
    lambda_step: float = 0.1
    model_opac_dir: str = "COM/contopac"

    # Grid Points
    # Format: [[Teff, logg, Fe/H, microturb_str], ...]
    grid_points: List[Tuple] = field(default_factory=list)

    def __post_init__(self):
        # Set derived paths if not provided
        if not self.model_atmosphere_path:
            self.model_atmosphere_path = os.path.join(self.project_root, "input_files", "model_atmospheres", "1D", "marcs_standard_comp", "marcs_standard_comp")
        if not self.linelist_path:
            self.linelist_path = os.path.join(self.project_root, "input_files", "linelists")
        if self.linelist_files is None:
            # Default to the one we found or empty
            self.linelist_files = ["nlte_ges_linelist_jmg6may2025_I_II"]
        # Ensure NLTE info file has a default if NLTE is enabled
        if self.nlte and not self.nlte_info_file:
            self.nlte_info_file = os.path.join(self.project_root, "DATA", "SPECIES_LTE_NLTE.dat")
        if not self.output_dir:
            self.output_dir = os.path.join(self.project_root, "spectra")
        if not self.log_dir:
            self.log_dir = os.path.join(self.project_root, "logs")
        if not self.tmp_dir:
            self.tmp_dir = os.path.join(self.project_root, "tmp")
            
        # Set executable paths
        exec_dir = os.path.join(self.project_root, f"exec-{self.compiler}")
        self.babsma_path = os.path.join(exec_dir, self.babsma_exec)
        self.bsyn_path = os.path.join(exec_dir, self.bsyn_exec)
        # Interpolator is usually in a separate directory
        self.interpol_path = os.path.join(self.project_root, "interpolator", self.interpol_exec)

        print("\n--- Turbospectrum Configuration ---")
        for key, value in dataclasses.asdict(self).items():
            print(f"{key}: {value}")
        print("-----------------------------------")



# =============================================================================
# LOGIC
# =============================================================================

def ensure_directories(config: TurbospectrumConfig):
    for path in [config.output_dir, config.log_dir, config.tmp_dir]:
        os.makedirs(path, exist_ok=True)
    # Also ensure opac dir exists
    opac_full_path = os.path.join(config.project_root, config.model_opac_dir)
    os.makedirs(opac_full_path, exist_ok=True)

def create_linelist_file(config: TurbospectrumConfig) -> str:
    """Creates a file containing the list of linelists to use."""
    list_file_path = os.path.join(config.tmp_dir, "linelists.txt")
    with open(list_file_path, "w") as f:
        for linelist in config.linelist_files:
            # If it's an absolute path, use it. Otherwise join with linelist_path
            if os.path.isabs(linelist):
                path = linelist
            else:
                path = os.path.join(config.linelist_path, linelist)
            f.write(f"{path}\n") # Turbospectrum does not want quotes in the list file apparently
            
    return list_file_path

def get_model_filename(teff, logg, feh, turb_str):
    # Construct the model filename based on the convention
    # Example: p2500_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod
    
    logg_str = f"{logg:+.1f}" # Note: +3.0 not +3.00
    feh_str = f"{feh:+.2f}"
    
    filename = f"p{teff}_g{logg_str}_m0.0_t{turb_str}_st_z{feh_str}_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    return filename

class ModelInterpolator:
    def __init__(self, config: TurbospectrumConfig):
        self.config = config
        self.available_models = []
        self._scan_models()

    def _scan_models(self):
        """Scans the model directory and parses filenames."""
        pattern = os.path.join(self.config.model_atmosphere_path, "*.mod")
        files = glob.glob(pattern)
        
        # Regex to parse filename
        # p2500_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod
        # We need to extract Teff, logg, FeH, and keep track of other params (turb, alpha, etc) to match
        # Assuming standard format
        regex = re.compile(r"p(\d+)_g([+\-]\d+\.\d+)_m0\.0_t(\d+)_st_z([+\-]\d+\.\d+)_a([+\-]\d+\.\d+)_.*\.mod")
        
        self.available_models = []
        for f in files:
            basename = os.path.basename(f)
            match = regex.match(basename)
            if match:
                teff = int(match.group(1))
                logg = float(match.group(2))
                turb = match.group(3)
                feh = float(match.group(4))
                alpha = float(match.group(5))
                
                self.available_models.append({
                    'teff': teff,
                    'logg': logg,
                    'feh': feh,
                    'turb': turb,
                    'alpha': alpha,
                    'path': f,
                    'filename': basename
                })

    def find_bracketing_models(self, target_teff, target_logg, target_feh, target_turb):
        """Finds the 8 bracketing models for interpolation."""
        # Filter by turbulence (must match)
        # We assume alpha is 0.0 for now or matches target if we had target alpha
        candidates = [m for m in self.available_models if m['turb'] == target_turb]
        
        if not candidates:
            return None, "No models found with matching turbulence"

        # Get unique grid points
        teffs = sorted(list(set(m['teff'] for m in candidates)))
        loggs = sorted(list(set(m['logg'] for m in candidates)))
        fehs = sorted(list(set(m['feh'] for m in candidates)))
        
        # Helper to find bracket
        def get_bracket(values, target):
            values = sorted(values)
            if target <= values[0]: return values[0], values[1]
            if target >= values[-1]: return values[-2], values[-1]
            for i in range(len(values)-1):
                if values[i] <= target < values[i+1]:
                    return values[i], values[i+1]
            return values[0], values[1] # Should not happen

        t1, t2 = get_bracket(teffs, target_teff)
        g1, g2 = get_bracket(loggs, target_logg)
        z1, z2 = get_bracket(fehs, target_feh)
        
        # Construct the 8 combinations
        # Order matters for interpol_modeles? 
        # The shell script does:
        # (t1, g1, z1), (t1, g1, z2), (t1, g2, z1), (t1, g2, z2)
        # (t2, g1, z1), (t2, g1, z2), (t2, g2, z1), (t2, g2, z2)
        
        brackets = []
        for t in [t1, t2]:
            for g in [g1, g2]:
                for z in [z1, z2]:
                    # Find the specific model file
                    match = next((m for m in candidates if m['teff'] == t and abs(m['logg'] - g) < 0.01 and abs(m['feh'] - z) < 0.01), None)
                    if not match:
                        return None, f"Missing grid point: Teff={t}, logg={g}, FeH={z}"
                    brackets.append(match['path'])
                    
        return brackets, None

    def interpolate(self, teff, logg, feh, turb_str, output_path):
        """Runs the interpolation."""
        brackets, error = self.find_bracketing_models(teff, logg, feh, turb_str)
        if not brackets:
            return False, error
            
        # Prepare input for interpol_modeles
        # Input format:
        # 'model1'
        # ...
        # 'model8'
        # 'output_model'
        # 'output_alt'
        # teff
        # logg
        # feh
        # .false.
        # .false.
        # ''
        
        input_str = ""
        for b in brackets:
            input_str += f"'{b}'\n"
            
        alt_path = os.path.join(self.config.tmp_dir, os.path.basename(output_path) + ".alt")
        
        input_str += f"'{output_path}'\n"
        input_str += f"'{alt_path}'\n"
        input_str += f"{teff}\n"
        input_str += f"{logg}\n"
        input_str += f"{feh}\n"
        input_str += ".false.\n" # optimize?
        input_str += ".false.\n" # some other flag?
        input_str += "''\n"
        
        try:
            process = subprocess.run(
                [self.config.interpol_path],
                input=input_str,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.config.project_root
            )
            if process.returncode != 0:
                return False, f"Interpolation failed:\n{process.stdout}"
        except Exception as e:
            return False, f"Interpolation execution error: {e}"
            
        return True, "Success"

def run_single_synthesis(args):
    params, config = args
    teff, logg, feh, turb_str = params
    
    # Map turb_str to float for babsma input if needed
    # Assuming "01" -> 1.0, "02" -> 2.0 etc.
    try:
        turb_val = float(turb_str) # This might be wrong if it's just an ID, but usually t01 = 1km/s
        if turb_val > 10: # e.g. if it was 10 meaning 1.0? Unlikely.
             turb_val = turb_val / 10.0
    except:
        turb_val = 1.0 # Default fallback

    model_file = get_model_filename(teff, logg, feh, turb_str)
    model_path = os.path.join(config.model_atmosphere_path, model_file)
    
    base_name = os.path.splitext(model_file)[0]
    log_file = os.path.join(config.log_dir, f"{base_name}.log")
    opac_path = os.path.join(config.project_root, config.model_opac_dir, f"{base_name}opac")
    result_file = os.path.join(config.output_dir, f"{base_name}.spec")
    
    # Check if output exists and skip if force is False
    expected_outputs = []
    if config.calculate_intensity:
        expected_outputs.append(os.path.join(config.output_dir, f"{base_name}.intensity.spec"))
    else:
        expected_outputs.append(os.path.join(config.output_dir, f"{base_name}.spec"))

    if not config.force:
        all_exist = True
        for f in expected_outputs:
            if not os.path.exists(f):
                all_exist = False
                break
        if all_exist:
            return f"WARNING: Skipping {base_name}, output exists."
    
    # Check if model exists, if not try to interpolate
    if not os.path.exists(model_path):
        # Initialize interpolator (this might be expensive to do every time, but safe for multiprocessing)
        # Alternatively, pass an initialized interpolator if it was picklable (files list might be large?)
        # Scanning directory is fast enough.
        interpolator = ModelInterpolator(config)
        success, message = interpolator.interpolate(teff, logg, feh, turb_str, model_path)
        if not success:
            return f"ERROR: Model not found and interpolation failed: {model_path}\nReason: {message}"
        else:
            # Log interpolation success?
            pass

    # Check if model is a standard MARCS model or interpolated
    is_marcs = True
    try:
        with open(model_path, 'r') as f:
            first_line = f.readline()
            if "INTERPOL" in first_line:
                is_marcs = False
    except:
        pass # Assume MARCS if read fails? Or fail later.

    marcs_flag = '.true.' if is_marcs else '.false.'

    with open(log_file, "w") as log:
        log.write(f"Starting synthesis for {base_name}\n")
        
        # ---------------------------------------------------------------------
        # Step 1: BABSMA (Continuous Opacity)
        # ---------------------------------------------------------------------
        babsma_input = f"""'LAMBDA_MIN:'  '{config.lambda_min}'
'LAMBDA_MAX:'  '{config.lambda_max}'
'LAMBDA_STEP:' '{config.lambda_step}'
'MODELINPUT:' '{model_path}'
'MARCS-FILE:' '{marcs_flag}'
'MODELOPAC:' '{opac_path}'
'ABUND_SOURCE:' 'magg'
'METALLICITY:'    '{feh}'
'ALPHA/Fe   :'    '0.00'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:'   '0'
'XIFIX:' 'T'
{turb_val}
"""
        log.write("\n--- BABSMA INPUT ---\n")
        log.write(babsma_input)
        log.write("\n--------------------\n")
        
        try:
            process = subprocess.run(
                [config.babsma_path],
                input=babsma_input,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=config.project_root # Run from root so relative paths in Fortran work if needed
            )
            if process.returncode != 0:
                return f"ERROR: babsma failed for {base_name}"
        except Exception as e:
            return f"EXCEPTION: babsma execution failed: {e}"

        # ---------------------------------------------------------------------
        # Step 2: BSYN (Spectral Synthesis)
        # ---------------------------------------------------------------------
        
        # Determine synthesis mode
        # If calculate_intensity is True, we run for Intensity. 
        # Note: For Plane-Parallel models, Turbospectrum outputs intensities for 12 standard mu angles 
        # in a single file, so we don't need to loop over angles.
        
        synthesis_runs = []
        if config.calculate_intensity:
            synthesis_runs.append({
                'mode': 'Intensity',
                'suffix': ".intensity"
            })
        else:
            # Default Flux calculation
            synthesis_runs.append({
                'mode': 'Flux',
                'suffix': ""
            })

        for run in synthesis_runs:
            mode_str = run['mode']
            suffix = run['suffix']
            
            current_result_file = os.path.join(config.output_dir, f"{base_name}{suffix}.spec")
            
            bsyn_input = f"""'NLTE :'          '{'.true.' if config.nlte else '.false.'}'
'NLTEINFOFILE:'  '{config.nlte_info_file if config.nlte_info_file else 'DATA/SPECIES_LTE_NLTE.dat'}'
'LAMBDA_MIN:'     '{config.lambda_min}'
'LAMBDA_MAX:'     '{config.lambda_max}'
'LAMBDA_STEP:'    '{config.lambda_step}'
'INTENSITY/FLUX:' '{mode_str}'
'MODELOPAC:' '{opac_path}'
'RESULTFILE :' '{current_result_file}'
'ABUND_SOURCE:'   'magg'
'METALLICITY:'    '{feh}'
'ALPHA/Fe   :'    '0.00'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:'   '0'
'ISOTOPES : ' '0'
'LIST_OF_LINELISTS:' '{config.linelist_file_path}'
'SPHERICAL:'  'F'
  30
  300.00
  15
  {turb_val:.2f}
"""
            log.write(f"\n--- BSYN INPUT ({mode_str}) ---\n")
            log.write(bsyn_input)
            log.write("\n------------------\n")

            try:
                process = subprocess.run(
                    [config.bsyn_path],
                    input=bsyn_input,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=config.project_root
                )
                if process.returncode != 0:
                    return f"ERROR: bsyn failed for {base_name} ({mode_str})"
            except Exception as e:
                return f"EXCEPTION: bsyn execution failed: {e}"

    return f"SUCCESS: {base_name}"

def run_grid(config: TurbospectrumConfig, grid_points: List[Tuple]):
    """
    Runs the Turbospectrum synthesis for a given configuration and list of grid points.
    """
    ensure_directories(config)
    
    # Create the linelist file once
    config.linelist_file_path = create_linelist_file(config)
    
    print(f"Running Turbospectrum in {config.project_root}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of grid points: {len(grid_points)}")
    
    # Prepare arguments for parallel execution
    # We pass config to each worker
    tasks = [(point, config) for point in grid_points]
    
    # Determine number of CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPUs")
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_synthesis, tasks)
        
    end_time = time.time()
    
    # Report results
    print("\n--- Summary ---")
    success_count = 0
    for res in results:
        print(res)
        if res.startswith("SUCCESS"):
            success_count += 1
            
    print(f"\nCompleted {success_count}/{len(grid_points)} calculations in {end_time - start_time:.2f} seconds.")

def main():
    # Detect project root (assuming this script is in scripts/ or root)
    # Adjust this logic if you move the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    
    # Parse arguments manually to handle --force and config file
    args = sys.argv[1:]
    force_flag = False
    if "--force" in args:
        force_flag = True
        args.remove("--force")
    
    # Load configuration from JSON file if provided as argument
    if len(args) > 0:
        config_path = args[0]
        with open(config_path, 'r') as f:
            cfg_data = json.load(f)
        if 'project_root' not in cfg_data:
            cfg_data['project_root'] = project_root
        config = TurbospectrumConfig(**cfg_data)
    else:
        config = TurbospectrumConfig(project_root=project_root)
    
    # Apply force flag
    if force_flag:
        config.force = True
    
    # Example: Enable intensity calculation
    # config.calculate_intensity = True
    # config.mu_angles = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    run_grid(config, config.grid_points)

if __name__ == "__main__":
    main()
