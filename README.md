# TurbospectrumNLTE (aka Turbospectrum2020)
Public

Synthetic stellar spectra calculator LTE / NLTE

Bertrand Plez
LUPM, Montpellier University, France

in collaboration with Jeff Gerber, Ekaterina Magg, and Maria Bergemann

The next version of TS (Turbospectrum), with NLTE capabilities.
In order to compute NLTE stellar spectra, additional data is needed.
See documentation in DOC folder

A wrapper can be found there: 
https://github.com/EkaterinaSe/TurboSpectrum-Wrapper/
and another one there:
https://github.com/JGerbs13/TSFitPy

## Downloading Data

A script is provided to download the necessary data to run Turbospectrum. This includes model atmospheres, NLTE data, and line lists.

### Usage

To use the script, run it from the command line with the desired options. You can download specific parts of the data, or all of it at once.

First, make the script executable:
```bash
chmod +x scripts/download_data.sh
```

Then, run the script with one of the following options:

*   **Download all data:**
    ```bash
    ./scripts/download_data.sh --all
    ```

*   **Download specific model atmospheres:**
    ```bash
    # Download MARCS atmospheres
    ./scripts/download_data.sh --atmospheres MARCS

    # Download STAGGER atmospheres
    ./scripts/download_data.sh --atmospheres STAGGER
    ```

*   **Download NLTE data:**
    ```bash
    # Download all NLTE data
    ./scripts/download_data.sh --nlte-atoms all
    ```

*   **Download line lists:**
    ```bash
    ./scripts/download_data.sh --linelists
    ```

For more information, you can view the help message:
```bash
./scripts/download_data.sh --help
```

## Workflow

This section outlines the typical workflow for generating a grid of synthetic spectra.

### 1. Configure the Parameter Grid

The first step is to define the grid of stellar parameters for which you want to compute spectra. This is done by editing the `scripts/grid_config.yml` file.

This YAML file allows you to specify the minimum, maximum, and step size for `teff`, `logg`, and `feh`. You can also set the desired wavelength range and other synthesis parameters.

### 2. Generate the Parameter CSV

Once you have configured `grid_config.yml`, you can generate the `parameter_grid.csv` file by running the `generate_grid.py` script. This script requires the `PyYAML` and `numpy` packages to be installed so it can resolve lists, ranges, and sampled distributions.

```bash
# Install dependencies if you haven't already
pip install PyYAML numpy

# Run the script to generate the grid
python3 scripts/generate_grid.py
```

This will create the `scripts/parameter_grid.csv` file, which will be used by the subsequent scripts in the pipeline.

### 3. Interpolate Model Atmospheres

With the parameter grid generated, the next step is to ensure that a model atmosphere exists for each point in the grid. The `interpolate_models.sh` script handles this by interpolating new models from the existing grid as needed.

The script reads the `scripts/parameter_grid.csv` file and, for each entry, checks if the required model atmosphere exists. If not, it interpolates one.

To run the script:
```bash
./scripts/interpolate_models.sh
```

This will populate the `input_files/model_atmospheres/` directory with any newly interpolated models.

### 4. Synthesize a Grid of Spectra

With atmospheres and a parameter grid in place, you can generate spectra for every sampled point using `scripts/synthesize_spectra.sh`. Make sure `scripts/env.sh` points to your local paths for model atmospheres, line lists, and Turbospectrum executables.

```bash
# Generate a reproducible grid with sampling controls
python3 scripts/generate_grid.py

# Ensure atmospheres exist for each grid point
./scripts/interpolate_models.sh

# Synthesize Flux/Intensity spectra in parallel across the grid
./scripts/synthesize_spectra.sh
```

Each run reads `scripts/parameter_grid.csv` (including `grid_version`, abundances, and sampling metadata), uses the corresponding model file, and writes logs under `logs/` with filenames containing the grid version, atmosphere parameters, output mode, and calculation mode. Synthetic spectra are written to the directory specified by `SPECTRA_PATH` in `scripts/env.sh`.
