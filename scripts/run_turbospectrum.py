import os
import sys
import subprocess
import multiprocessing
import time
import glob
import re
import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict, Iterator, Mapping
import json
import csv
from datetime import datetime

try:
    from .nlte_ascii_departures import (
        NLTE_ASCII_CONTROL_KEYS,
        materialize_nlte_info_with_departure_override,
        resolve_absolute_abundance,
        select_departure_file,
        selector_from_row,
    )
except ImportError:
    from nlte_ascii_departures import (
        NLTE_ASCII_CONTROL_KEYS,
        materialize_nlte_info_with_departure_override,
        resolve_absolute_abundance,
        select_departure_file,
        selector_from_row,
    )

# =============================================================================
# CONFIGURATION
# =============================================================================

def _strip_private_keys(obj):
    """Recursively remove JSON keys starting with '_' (e.g. _comment, _note)."""
    if isinstance(obj, dict):
        return {
            k: _strip_private_keys(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, list):
        return [_strip_private_keys(v) for v in obj]
    return obj


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _normalize_output_mode(raw: Any) -> str:
    text = str(raw if raw is not None else "Flux").strip().lower()
    if text == "intensity":
        return "Intensity"
    if text == "flux":
        return "Flux"
    raise ValueError(f"output_mode must be 'Flux' or 'Intensity', got {raw!r}")


class LinelistValidationError(ValueError):
    """Raised when configured linelist files fail upfront validation."""


_TURBOSPECTRUM_ERROR_RE = re.compile(
    r"\b(error|failed|exception|segmentation|traceback|forrtl|abort|fatal|severe|cannot|can't)\b|^\s*stop\b",
    flags=re.IGNORECASE,
)


def _extract_turbospectrum_log_excerpt(
    log_file: str,
    *,
    max_lines: int = 12,
    max_chars: int = 1200,
) -> str:
    if not log_file or not os.path.isfile(log_file):
        return ""

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
            lines = [re.sub(r"\s+", " ", line).strip() for line in handle if line.strip()]
    except OSError as exc:
        return f"log unreadable: {exc}"

    if not lines:
        return ""

    interesting = [line for line in lines if _TURBOSPECTRUM_ERROR_RE.search(line)]
    excerpt_lines = interesting[-max_lines:] if interesting else lines[-max_lines:]
    excerpt = " | ".join(excerpt_lines)
    if len(excerpt) > max_chars:
        excerpt = f"{excerpt[: max_chars - 3]}..."
    return excerpt


def _with_turbospectrum_log_context(message: str, log_file: str) -> str:
    parts = [str(message).strip()]
    if log_file:
        parts.append(f"log={log_file}")
    excerpt = _extract_turbospectrum_log_excerpt(log_file)
    if excerpt:
        parts.append(f"excerpt={excerpt}")
    return "; ".join(part for part in parts if part)


def _normalize_config_dict(cfg_data: dict, default_project_root: str) -> dict:
    """
    Normalize config JSON to the flat schema expected by TurbospectrumConfig.

    Supports both:
    - legacy flat configs (already matching TurbospectrumConfig fields)
    - comprehensive nested configs (configs/synthesis/config_sample_comprehensive.json)
    """
    cfg_data = _strip_private_keys(cfg_data or {})

    has_nested_sections = any(
        key in cfg_data for key in ("paths", "executables", "synthesis_parameters", "nlte")
    )
    if not has_nested_sections:
        if not cfg_data.get("project_root"):
            cfg_data["project_root"] = default_project_root
        if "output_mode" not in cfg_data:
            if "calculate_intensity" in cfg_data or "compute_intensity" in cfg_data:
                cfg_data["output_mode"] = "Intensity" if _as_bool(
                    cfg_data.get("calculate_intensity", cfg_data.get("compute_intensity"))
                ) else "Flux"
            elif "intensity_flux" in cfg_data:
                cfg_data["output_mode"] = _normalize_output_mode(cfg_data.get("intensity_flux"))
            else:
                cfg_data["output_mode"] = "Flux"
        else:
            cfg_data["output_mode"] = _normalize_output_mode(cfg_data.get("output_mode"))
        cfg_data.pop("calculate_intensity", None)
        cfg_data.pop("compute_intensity", None)
        cfg_data.pop("intensity_flux", None)

        mu_sampling = cfg_data.get("mu_sampling", {}) or {}
        if not isinstance(mu_sampling, dict):
            mu_sampling = {}
        if cfg_data["output_mode"] == "Intensity":
            mode = str(mu_sampling.get("mode", "none")).strip().lower()
            if mode in {"", "none"}:
                mu_sampling["mode"] = "random"
        cfg_data["mu_sampling"] = mu_sampling
        return cfg_data

    flat: dict = {}
    cfg_project_root = cfg_data.get("project_root", default_project_root)
    if not cfg_project_root:
        cfg_project_root = default_project_root
    flat["project_root"] = cfg_project_root
    flat["compiler"] = cfg_data.get("compiler", "gf")
    flat["force"] = cfg_data.get("force", False)
    flat["max_workers"] = cfg_data.get("max_workers", None)

    executables = cfg_data.get("executables", {}) or {}
    flat["babsma_exec"] = executables.get("babsma_exec", "babsma_lu")
    flat["bsyn_exec"] = executables.get("bsyn_exec", "bsyn_lu")
    flat["interpol_exec"] = executables.get("interpol_exec", "interpol_modeles")

    paths = cfg_data.get("paths", {}) or {}
    flat["model_atmosphere_path"] = paths.get("model_atmosphere_path", "")
    flat["linelist_path"] = paths.get("linelist_path", "")
    flat["linelist_files"] = paths.get("linelist_files", None)
    flat["output_dir"] = paths.get("output_dir", "")
    flat["log_dir"] = paths.get("log_dir", "")
    flat["tmp_dir"] = paths.get("tmp_dir", "")
    flat["model_opac_dir"] = paths.get("model_opac_dir", "COM/contopac")

    synthesis = cfg_data.get("synthesis_parameters", {}) or {}
    flat["lambda_min"] = synthesis.get("lambda_min", 4000.0)
    flat["lambda_max"] = synthesis.get("lambda_max", 8000.0)
    flat["lambda_step"] = synthesis.get("lambda_step", 0.1)
    # BSYN control parameter (historically hardcoded to 300.00 in shell/python runners).
    # Accept explicit config override when present.
    flat["resolution"] = synthesis.get("resolution", cfg_data.get("resolution", 300.0))
    if "output_mode" in synthesis:
        output_mode = _normalize_output_mode(synthesis.get("output_mode"))
    elif "intensity_flux" in synthesis:
        output_mode = _normalize_output_mode(synthesis.get("intensity_flux"))
    else:
        output_mode = "Intensity" if _as_bool(
            synthesis.get("calculate_intensity", synthesis.get("compute_intensity", False))
        ) else "Flux"
    flat["output_mode"] = output_mode
    flat["mu_angles"] = synthesis.get("mu_angles", []) or []
    # Optional: choose a subset of mu points from an Intensity spectrum.
    # This is used by the Zarr-grid synthesis scripts when reading Turbospectrum outputs.
    mu_sampling = synthesis.get("mu_sampling", {}) or {}
    if not isinstance(mu_sampling, dict):
        mu_sampling = {}
    if output_mode == "Intensity":
        mode = str(mu_sampling.get("mode", "none")).strip().lower()
        if mode in {"", "none"}:
            mu_sampling["mode"] = "random"
    flat["mu_sampling"] = mu_sampling

    nlte_cfg = cfg_data.get("nlte", {}) or {}
    flat["nlte"] = nlte_cfg.get("enabled", False)
    flat["nlte_info_file"] = nlte_cfg.get("nlte_info_file", "")

    provenance_cfg = cfg_data.get("provenance", {}) or {}
    model_atmosphere_cfg = cfg_data.get("model_atmosphere", {}) or {}

    def _first_nonempty(*values, default: str = ""):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
                continue
            return value
        return default

    geometry_from_model_cfg = model_atmosphere_cfg.get("geometry")
    if geometry_from_model_cfg in (None, "") and "spherical" in model_atmosphere_cfg:
        try:
            geometry_from_model_cfg = "spherical" if bool(model_atmosphere_cfg.get("spherical")) else "plane_parallel"
        except Exception:
            geometry_from_model_cfg = ""

    # Prefer explicit provenance overrides, then direct config fields, then nearby
    # sections (paths/model_atmosphere) to avoid duplicate provenance declarations.
    flat["linelist_version"] = _first_nonempty(
        provenance_cfg.get("linelist_version"),
        cfg_data.get("linelist_version"),
        paths.get("linelist_version"),
    )
    flat["linelist_sha256"] = _first_nonempty(
        provenance_cfg.get("linelist_sha256"),
        cfg_data.get("linelist_sha256"),
        paths.get("linelist_sha256"),
        paths.get("linelist_checksum"),
    )
    flat["linelist_preprocessing"] = _first_nonempty(
        provenance_cfg.get("linelist_preprocessing"),
        cfg_data.get("linelist_preprocessing"),
        paths.get("linelist_preprocessing"),
    )
    flat["atmosphere_geometry"] = _first_nonempty(
        provenance_cfg.get("atmosphere_geometry"),
        cfg_data.get("atmosphere_geometry"),
        geometry_from_model_cfg,
    )
    flat["atmosphere_version"] = _first_nonempty(
        provenance_cfg.get("atmosphere_version"),
        cfg_data.get("atmosphere_version"),
        model_atmosphere_cfg.get("version"),
        paths.get("model_atmosphere_version"),
    )
    flat["atmosphere_sha256"] = _first_nonempty(
        provenance_cfg.get("atmosphere_sha256"),
        cfg_data.get("atmosphere_sha256"),
        model_atmosphere_cfg.get("sha256"),
        paths.get("model_atmosphere_sha256"),
    )
    flat["synthesis_code_version"] = _first_nonempty(
        provenance_cfg.get("synthesis_code_version"),
        cfg_data.get("synthesis_code_version"),
    )
    flat["spice_version"] = _first_nonempty(
        provenance_cfg.get("spice_version"),
        cfg_data.get("spice_version"),
    )
    flat["environment_capture"] = _first_nonempty(
        provenance_cfg.get("environment_capture"),
        cfg_data.get("environment_capture"),
        default="pip_freeze",
    )

    # Grid points: accept either legacy [[teff, logg, feh, turb_str], ...]
    # or comprehensive objects [{teff, logg, feh, microturb_str}, ...]
    grid_points = cfg_data.get("grid_points", []) or []
    normalized_points: List[Tuple] = []
    for point in grid_points:
        if isinstance(point, (list, tuple)) and len(point) >= 4:
            try:
                normalized_points.append((int(point[0]), float(point[1]), float(point[2]), str(point[3]).strip()))
            except Exception:
                continue
        elif isinstance(point, dict):
            if "teff" not in point or "logg" not in point or "feh" not in point:
                continue
            try:
                teff = int(point["teff"])
                logg = float(point["logg"])
                feh = float(point["feh"])
            except Exception:
                continue
            turb_str = point.get("microturb_str") or point.get("t_value") or point.get("turb") or "01"
            normalized_points.append((teff, logg, feh, str(turb_str).strip()))

    flat["grid_points"] = normalized_points
    flat["grid_points_file"] = cfg_data.get("grid_points_file", "")

    return flat


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

    # Output mode
    output_mode: str = "Flux"
    mu_angles: List[float] = field(default_factory=list)
    # Optional post-processing configuration for intensity outputs.
    # Expected shape (all keys optional):
    #   {"mode": "nearest", "count": 1, "seed": 123, "min": 0.0, "max": 1.0, "reduce": "first"}
    mu_sampling: Dict[str, Any] = field(default_factory=dict)

    # Synthesis Parameters
    lambda_min: float = 4000
    lambda_max: float = 8000
    lambda_step: float = 0.1
    resolution: float = 300.0
    model_opac_dir: str = "COM/contopac"
    # Optional provenance metadata used to populate DATA_SCHEMA.md manifests.
    linelist_version: str = ""
    linelist_sha256: str = ""
    linelist_preprocessing: str = ""
    atmosphere_geometry: str = ""
    atmosphere_version: str = ""
    atmosphere_sha256: str = ""
    synthesis_code_version: str = ""
    spice_version: str = ""
    # Controls provenance environment capture; options: pip_freeze, conda_env_export, auto.
    environment_capture: str = "pip_freeze"

    # Parallelization
    # If None, the script will try to detect the assigned CPUs from the environment
    max_workers: Optional[int] = None

    # Grid Points
    # Format: [[Teff, logg, Fe/H, microturb_str], ...]
    grid_points: List[Tuple] = field(default_factory=list)
    
    # External grid points file (CSV format)
    # If specified, grid_points will be loaded from this file instead
    # CSV should have columns: teff, logg, feh, t_value (or similar)
    # This is optimized for very large grids to avoid loading all points into memory
    grid_points_file: str = ""

    def __post_init__(self):
        self.output_mode = _normalize_output_mode(self.output_mode)
        try:
            self.resolution = float(self.resolution)
        except Exception as exc:
            raise ValueError(f"resolution must be numeric, got {self.resolution!r}") from exc
        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution!r}")
        if not isinstance(self.mu_sampling, dict):
            self.mu_sampling = {}
        if self.output_mode == "Intensity":
            mode = str(self.mu_sampling.get("mode", "none")).strip().lower()
            if mode in {"", "none"}:
                self.mu_sampling["mode"] = "random"

        # Set derived paths if not provided
        if not self.model_atmosphere_path:
            # Prefer the common layout:
            #   input_files/model_atmospheres/1D/marcs_standard_comp/*.mod
            # but support older layouts with an extra nested marcs_standard_comp/
            candidate = os.path.join(
                self.project_root, "input_files", "model_atmospheres", "1D", "marcs_standard_comp"
            )
            nested = os.path.join(candidate, "marcs_standard_comp")
            self.model_atmosphere_path = nested if os.path.isdir(nested) else candidate
        if not self.linelist_path:
            self.linelist_path = os.path.join(self.project_root, "input_files", "linelists")
        if self.linelist_files is None:
            # Default to the one we found or empty
            self.linelist_files = ["nlte_ges_linelist_jmg6may2025_I_II"]
        # Ensure NLTE info file has a default if NLTE is enabled
        if self.nlte and not self.nlte_info_file:
            self.nlte_info_file = os.path.join(self.project_root, "DATA", "SPECIES_LTE_NLTE.dat")
        default_run_root = os.path.join(self.project_root, "runs", "local-dev")
        if not self.output_dir:
            self.output_dir = os.path.join(default_run_root, "outputs", "spectra")
        if not self.log_dir:
            self.log_dir = os.path.join(default_run_root, "logs", "shards")
        if not self.tmp_dir:
            self.tmp_dir = os.path.join(default_run_root, "tmp")
            
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

def load_grid_points_from_csv(file_path: str, project_root: str = "") -> Iterator[Tuple]:
    """
    Load grid points from a CSV file in a memory-efficient way.
    
    Args:
        file_path: Path to CSV file (can be relative to project_root or absolute)
        project_root: Project root directory for resolving relative paths
        
    Yields:
        Tuples of (teff, logg, feh, t_value) for each grid point
        
    CSV Format:
        Expected columns: teff, logg, feh, and one of: turbvel, t_value, turbulence, microturb
        The CSV can have additional columns which will be ignored.
        Column matching is case-insensitive and flexible (e.g., 'feh' or 'fe_h' or 'metallicity').
    """
    # Resolve path
    if not os.path.isabs(file_path):
        if project_root:
            file_path = os.path.join(project_root, file_path)
        else:
            # Try relative to current working directory
            pass
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Grid points file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Try to find the right column names (case-insensitive, flexible)
        # Map internal names to possible CSV column names
        col_mapping = {}
        
        # Map each required field to possible column names
        field_mappings = {
            'teff': ['teff'],
            'logg': ['logg'],
            'feh': ['feh', 'fe_h', 'metallicity'],
            'turb': ['turbvel', 't_value', 'turbulence', 'microturb']
        }
        
        for internal_name, possible_names in field_mappings.items():
            found = False
            # Try exact match first (case-sensitive)
            for possible_name in possible_names:
                if possible_name in reader.fieldnames:
                    col_mapping[internal_name] = possible_name
                    found = True
                    break
            
            # Try case-insensitive match if not found
            if not found:
                for possible_name in possible_names:
                    for col in reader.fieldnames:
                        if col.lower() == possible_name.lower():
                            col_mapping[internal_name] = col
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                raise ValueError(f"Required column for '{internal_name}' not found in CSV file. "
                               f"Tried: {possible_names}. Available columns: {reader.fieldnames}")
        
        for row in reader:
            try:
                teff = int(float(row[col_mapping['teff']]))
                logg = float(row[col_mapping['logg']])
                feh = float(row[col_mapping['feh']])
                t_value = str(row[col_mapping['turb']]).strip()
                
                # Format t_value to match expected format (e.g., "01", "02")
                # Remove leading zeros if needed, but keep as string
                if t_value.isdigit():
                    # Ensure it's zero-padded to 2 digits if it's a number
                    t_value = f"{int(t_value):02d}"
                
                yield (teff, logg, feh, t_value)
            except (ValueError, KeyError) as e:
                # Skip invalid rows but warn
                print(f"WARNING: Skipping invalid row in grid points file: {row}. Error: {e}")
                continue

def ensure_directories(config: TurbospectrumConfig):
    for path in [config.output_dir, config.log_dir, config.tmp_dir]:
        os.makedirs(path, exist_ok=True)
    # Also ensure opac dir exists
    opac_full_path = os.path.join(config.project_root, config.model_opac_dir)
    os.makedirs(opac_full_path, exist_ok=True)


def validate_runtime_environment(config: TurbospectrumConfig) -> None:
    """Fail fast when the Turbospectrum runtime is incomplete."""
    problems: List[str] = []

    required_execs = [
        ("babsma", getattr(config, "babsma_path", "")),
        ("bsyn", getattr(config, "bsyn_path", "")),
        ("interpolator", getattr(config, "interpol_path", "")),
    ]
    for label, path in required_execs:
        resolved = str(path or "").strip()
        if not resolved:
            problems.append(f"{label} executable path is empty")
            continue
        if not os.path.isfile(resolved):
            problems.append(f"{label} executable not found: {resolved}")
            continue
        if not os.access(resolved, os.X_OK):
            problems.append(f"{label} executable is not runnable: {resolved}")

    model_root = os.path.abspath(str(config.model_atmosphere_path or "").strip())
    if not model_root:
        problems.append("model_atmosphere_path is empty")
    elif not os.path.isdir(model_root):
        problems.append(f"model atmosphere directory not found: {model_root}")
    else:
        try:
            if not glob.glob(os.path.join(model_root, "*.mod")):
                problems.append(f"model atmosphere directory contains no .mod files: {model_root}")
        except OSError as exc:
            problems.append(f"could not inspect model atmosphere directory {model_root}: {exc}")

    if config.nlte and config.nlte_info_file:
        nlte_info = os.path.abspath(str(config.nlte_info_file))
        if not os.path.isfile(nlte_info):
            problems.append(f"NLTE info file not found: {nlte_info}")

    if problems:
        bullet_list = "\n".join(f"- {problem}" for problem in problems)
        raise FileNotFoundError(
            "Invalid Turbospectrum runtime configuration. "
            "Fix the missing binaries/data paths before rerunning:\n"
            f"{bullet_list}"
        )


def _parse_int_env(var_name: str) -> Optional[int]:
    """Parse integer-like environment variables robustly."""
    raw_value = os.environ.get(var_name)
    if not raw_value:
        return None

    match = re.search(r"\d+", raw_value)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None


def determine_worker_count(config: TurbospectrumConfig) -> int:
    """Determine how many worker processes to spawn on HPC systems."""
    # Prefer explicit configuration
    configured = config.max_workers if config.max_workers and config.max_workers > 0 else None

    # Respect SLURM, PBS, and similar schedulers
    scheduler_env_vars = [
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
        "OMP_NUM_THREADS",
    ]
    env_value = None
    env_source = None
    for var in scheduler_env_vars:
        env_val = _parse_int_env(var)
        if env_val:
            env_value = env_val
            env_source = var
            break

    # Respect cgroup/affinity limits if present (common on HPC nodes)
    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except Exception:
            affinity_count = None

    # Fallback to Python's view of the system
    host_cpu_count = multiprocessing.cpu_count()

    # Start with the most specific value available
    worker_count = configured or env_value or host_cpu_count

    # Never exceed affinity limits
    if affinity_count:
        worker_count = min(worker_count, affinity_count)

    worker_count = max(1, worker_count)

    print("Parallelization setup:")
    if configured:
        print(f"  Using user-configured max_workers={configured}")
    if env_source:
        print(f"  Detected {env_value} CPUs from {env_source}")
    if affinity_count:
        print(f"  CPU affinity allows {affinity_count} workers")
    print(f"  Host reports {host_cpu_count} CPUs")
    print(f"  Spawning {worker_count} worker processes")

    return worker_count

def resolve_linelist_paths(linelist_path: str, linelist_files: Optional[List[str]]) -> List[str]:
    """Resolve linelist entries into absolute file paths.

    Supports explicit files, glob patterns such as ``dir/*``, and bare
    directories, which are expanded to all files in that directory.
    """
    out: List[str] = []
    seen: set[str] = set()
    base_dir = str(linelist_path or "")

    for item in (linelist_files or []):
        raw = str(item).strip()
        if not raw:
            continue

        candidate = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(base_dir, raw))

        if glob.has_magic(candidate):
            matches = sorted(os.path.abspath(path) for path in glob.glob(candidate))
            paths = [path for path in matches if os.path.isfile(path)]
            if not paths:
                raise FileNotFoundError(f"linelist pattern {raw!r} matched no files")
        elif os.path.isdir(candidate):
            matches = sorted(
                os.path.abspath(os.path.join(candidate, name))
                for name in os.listdir(candidate)
                if not name.startswith(".")
            )
            paths = [path for path in matches if os.path.isfile(path)]
            if not paths:
                raise FileNotFoundError(f"linelist directory {raw!r} contained no files")
        else:
            paths = [candidate]

        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            out.append(path)

    return out


def validate_linelist_files(linelist_path: str, linelist_files: Optional[List[str]]) -> List[str]:
    """Resolve and sanity-check linelist files before launching synthesis."""
    try:
        resolved = resolve_linelist_paths(linelist_path, linelist_files)
    except FileNotFoundError as exc:
        raise LinelistValidationError(f"Invalid linelist configuration: {exc}") from exc

    if not resolved:
        raise LinelistValidationError(
            "Invalid linelist configuration: no linelist files were resolved from linelist_files"
        )

    issues: List[str] = []
    for path in resolved:
        if not os.path.exists(path):
            issues.append(f"{path}: file does not exist")
            continue
        if not os.path.isfile(path):
            issues.append(f"{path}: not a regular file")
            continue
        if not os.access(path, os.R_OK):
            issues.append(f"{path}: file is not readable")
            continue

        try:
            size_bytes = os.path.getsize(path)
        except OSError as exc:
            issues.append(f"{path}: could not read file metadata ({exc})")
            continue
        if size_bytes <= 0:
            issues.append(f"{path}: file is empty and appears corrupted")
            continue

        try:
            with open(path, "rb") as handle:
                sample = handle.read(4096)
        except OSError as exc:
            issues.append(f"{path}: could not read file contents ({exc})")
            continue

        if not sample:
            issues.append(f"{path}: file is empty and appears corrupted")
            continue
        if b"\x00" in sample:
            issues.append(f"{path}: file contains NUL bytes and appears corrupted")

    if issues:
        details = "\n".join(f" - {issue}" for issue in issues)
        raise LinelistValidationError(
            "Invalid linelist input. Fix linelist_path/linelist_files before rerunning:\n"
            f"{details}"
        )

    return resolved

def create_linelist_file(config: TurbospectrumConfig) -> str:
    """Creates a file containing the list of linelists to use."""
    list_file_path = os.path.join(config.tmp_dir, "linelists.txt")
    with open(list_file_path, "w") as f:
        for path in validate_linelist_files(config.linelist_path, config.linelist_files):
            f.write(f"{path}\n") # Turbospectrum does not want quotes in the list file apparently
            
    return list_file_path

def get_model_filename(teff, logg, feh, turb_str):
    # Construct the model filename based on the convention
    # Example: p2500_g+3.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod
    
    logg_str = f"{logg:+.1f}" # Note: +3.0 not +3.00
    feh_str = f"{feh:+.2f}"
    
    filename = f"p{teff}_g{logg_str}_m0.0_t{turb_str}_st_z{feh_str}_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    return filename


PERIODIC_TABLE = {
    "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8, "f": 9, "ne": 10,
    "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15, "s": 16, "cl": 17, "ar": 18, "k": 19, "ca": 20,
    "sc": 21, "ti": 22, "v": 23, "cr": 24, "mn": 25, "fe": 26, "co": 27, "ni": 28, "cu": 29, "zn": 30,
    "ga": 31, "ge": 32, "as": 33, "se": 34, "br": 35, "kr": 36, "rb": 37, "sr": 38, "y": 39, "zr": 40,
    "nb": 41, "mo": 42, "tc": 43, "ru": 44, "rh": 45, "pd": 46, "ag": 47, "cd": 48, "in": 49, "sn": 50,
    "sb": 51, "te": 52, "i": 53, "xe": 54, "cs": 55, "ba": 56, "la": 57, "ce": 58, "pr": 59, "nd": 60,
    "pm": 61, "sm": 62, "eu": 63, "gd": 64, "tb": 65, "dy": 66, "ho": 67, "er": 68, "tm": 69, "yb": 70,
    "lu": 71, "hf": 72, "ta": 73, "w": 74, "re": 75, "os": 76, "ir": 77, "pt": 78, "au": 79, "hg": 80,
    "tl": 81, "pb": 82, "bi": 83, "po": 84, "at": 85, "rn": 86, "fr": 87, "ra": 88, "ac": 89, "th": 90,
    "pa": 91, "u": 92,
}

SPECIAL_ABUNDANCE_ALIASES = {
    "a": "a",
    "alpha": "a",
    "afe": "a",
    "alphafe": "a",
    "r": "r",
    "rfe": "r",
    "rprocess": "r",
    "rproc": "r",
    "s": "s",
    "sfe": "s",
    "sprocess": "s",
    "sproc": "s",
    "c": "c",
    "cfe": "c",
    "n": "n",
    "nfe": "n",
    "o": "o",
    "ofe": "o",
}

RESERVED_SYNTHESIS_KEYS = {
    "teff",
    "logg",
    "feh",
    "mu",
    "turb",
    "turbvel",
    "t_value",
    "microturb",
    "microturb_str",
    "lam_min",
    "lam_max",
    "lam_step",
    "output_mode",
    "calculation_mode",
    "mode",
    "grid_version",
    "row_index",
    "global_index",
    *NLTE_ASCII_CONTROL_KEYS,
}


def _normalize_abundance_key(raw_key: str) -> str:
    key = "".join(ch for ch in str(raw_key).strip().lower() if ch.isalnum())
    if not key:
        return ""
    if key in SPECIAL_ABUNDANCE_ALIASES:
        return SPECIAL_ABUNDANCE_ALIASES[key]
    if key.endswith("fe") and key[:-2] in PERIODIC_TABLE:
        return key[:-2]
    return key


def _format_abundance_value(raw_value: Any) -> str:
    try:
        return f"{float(raw_value):+0.2f}"
    except Exception:
        return str(raw_value).strip()


def _normalize_turbulence_id(raw_value: Any, default: str = "01") -> str:
    text = str(raw_value if raw_value is not None else "").strip()
    if not text:
        return default
    if text.isdigit():
        return f"{int(text):02d}"
    try:
        numeric = float(text)
    except Exception:
        return text
    if numeric.is_integer():
        return f"{int(numeric):02d}"
    return text


def _coerce_microturbulence_value(raw_value: Any, fallback: str) -> float:
    candidate = raw_value
    if candidate in (None, ""):
        candidate = fallback
    try:
        turb_val = float(str(candidate).strip())
        if turb_val > 10:
            turb_val = turb_val / 10.0
        return turb_val
    except Exception:
        return 1.0


def _collect_synthesis_abundances(params: Mapping[str, Any]) -> Dict[str, str]:
    abundances: Dict[str, str] = {}
    unsupported: List[str] = []
    for raw_key, raw_value in params.items():
        if raw_key in RESERVED_SYNTHESIS_KEYS:
            continue
        key = _normalize_abundance_key(str(raw_key))
        if not key:
            continue
        if key in {"a", "r", "s", "c", "n", "o"} or key in PERIODIC_TABLE:
            abundances[key] = _format_abundance_value(raw_value)
        else:
            unsupported.append(str(raw_key))
    if unsupported:
        raise ValueError(
            "Unsupported abundance column(s) for synthesis: "
            + ", ".join(sorted(unsupported))
        )
    return abundances


def _build_abundance_controls(abundances: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    normalized = {
        _normalize_abundance_key(name): _format_abundance_value(value)
        for name, value in abundances.items()
        if _normalize_abundance_key(name)
    }
    alpha = normalized.get("a", "+0.00")
    r_proc = normalized.get("r", "+0.00")
    s_proc = normalized.get("s", "+0.00")

    individual: Dict[str, str] = {}
    for key in ("c", "n", "o"):
        if key in normalized:
            individual[key] = normalized[key]
    for key, value in normalized.items():
        if key in {"a", "r", "s", "c", "n", "o"}:
            continue
        if key in PERIODIC_TABLE:
            individual[key] = value

    lines = [
        f"{PERIODIC_TABLE[key]:>3d}  {value}"
        for key, value in sorted(individual.items(), key=lambda item: PERIODIC_TABLE[item[0]])
    ]
    block = "\n".join(lines)
    if block:
        block += "\n"
    return alpha, r_proc, s_proc, block


def get_synthesis_stem(teff, logg, feh, turb_str, abundances: Optional[Mapping[str, Any]] = None):
    logg_str = f"{logg:+.1f}"
    feh_str = f"{feh:+.2f}"
    parts = [f"p{teff}", f"g{logg_str}", "m0.0", f"t{str(turb_str).strip()}", "st", f"z{feh_str}"]

    normalized = {
        _normalize_abundance_key(name): _format_abundance_value(value)
        for name, value in (abundances or {}).items()
        if _normalize_abundance_key(name)
    }
    legacy = {"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"}
    legacy.update({key: value for key, value in normalized.items() if key in legacy})
    parts.extend(f"{key}{legacy[key]}" for key in ("a", "c", "n", "o", "r", "s"))
    parts.extend(f"{key}{normalized[key]}" for key in sorted(key for key in normalized if key not in legacy))
    return "_".join(parts)


def _filename_token(raw_value: Any) -> str:
    text = str(raw_value if raw_value is not None else "").strip()
    if not text:
        return ""
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "+", "-"} else "-" for ch in text)
    return re.sub(r"-{2,}", "-", cleaned).strip("-")


def _resolve_synthesis_request(params: Mapping[str, Any]) -> Dict[str, Any]:
    teff = int(params["teff"])
    logg = float(params["logg"])
    feh = float(params["feh"])
    model_turb_str = _normalize_turbulence_id(
        params.get("microturb_str")
        or params.get("t_value")
        or params.get("turb")
        or params.get("turbvel")
        or "01"
    )
    synthesis_turb_raw = params.get("turbvel")
    if synthesis_turb_raw in (None, ""):
        synthesis_turb_raw = params.get("microturb")
    if synthesis_turb_raw in (None, ""):
        synthesis_turb_raw = params.get("turbulence")
    if synthesis_turb_raw in (None, ""):
        synthesis_turb_raw = model_turb_str
    abundances = _collect_synthesis_abundances(params)
    return {
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "model_turb_str": model_turb_str,
        "synthesis_turb_raw": synthesis_turb_raw,
        "abundances": abundances,
    }


def get_synthesis_output_stem_from_params(
    params: Mapping[str, Any],
    *,
    default_output_mode: str = "Flux",
    default_calculation_mode: str = "LTE",
    default_mode: str = "",
) -> str:
    request = _resolve_synthesis_request(params)
    base_stem = get_synthesis_stem(
        request["teff"],
        request["logg"],
        request["feh"],
        request["model_turb_str"],
        request["abundances"],
    )

    output_mode = params.get("output_mode")
    output_mode_text = str(output_mode if output_mode not in (None, "") else default_output_mode).strip()
    calculation_mode = params.get("calculation_mode")
    calculation_mode_text = str(
        calculation_mode if calculation_mode not in (None, "") else default_calculation_mode
    ).strip()
    mode = params.get("mode")
    mode_text = str(mode if mode not in (None, "") else default_mode).strip()

    extras: List[str] = []
    synthesis_turb_id = _normalize_turbulence_id(request["synthesis_turb_raw"], default=request["model_turb_str"])
    if synthesis_turb_id != request["model_turb_str"]:
        extras.append(f"xi{synthesis_turb_id}")

    lam_min = params.get("lam_min")
    lam_max = params.get("lam_max")
    lam_step = params.get("lam_step")
    if lam_min not in (None, "") and lam_max not in (None, "") and lam_step not in (None, ""):
        extras.append(
            "wl"
            + _filename_token(
                f"{float(lam_min):g}-{float(lam_max):g}-{float(lam_step):g}"
            )
        )

    if output_mode_text:
        extras.append(f"out{_filename_token(output_mode_text.lower())}")
    if calculation_mode_text:
        extras.append(f"calc{_filename_token(calculation_mode_text.lower())}")
    if mode_text:
        extras.append(f"mode{_filename_token(mode_text.lower())}")

    if not extras:
        return base_stem
    return "_".join([base_stem, *extras])


def _resolve_nlte_ascii_runtime_info(
    *,
    params: Mapping[str, Any],
    config: TurbospectrumConfig,
    model_input_path: str,
) -> Optional[Dict[str, Any]]:
    if not config.nlte:
        return None

    selector = selector_from_row(params)
    if selector is None:
        return None

    model_stem = os.path.splitext(os.path.basename(str(model_input_path)))[0]
    target_abundance, abundance_column = resolve_absolute_abundance(params, selector)
    selected_candidate = select_departure_file(
        directory=selector.directory,
        model_stem=model_stem,
        abundance=target_abundance,
        match=selector.match,
    )

    base_nlte_info_file = os.path.abspath(
        str(
            config.nlte_info_file
            if config.nlte_info_file
            else os.path.join(config.project_root, "DATA", "SPECIES_LTE_NLTE.dat")
        )
    )
    tmp_root = os.path.abspath(
        str(config.tmp_dir if config.tmp_dir else os.path.join(config.project_root, "tmp"))
    )
    runtime_nlte_info_file = materialize_nlte_info_with_departure_override(
        base_info_path=base_nlte_info_file,
        selector=selector,
        departure_file_path=selected_candidate.path,
        output_root=tmp_root,
    )

    return {
        "nlte_info_file": runtime_nlte_info_file,
        "departure_file": selected_candidate.path,
        "target_abundance": float(target_abundance),
        "matched_abundance": float(selected_candidate.abundance),
        "abundance_column": abundance_column,
        "species": selector.species,
        "model_stem": model_stem,
    }

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

    def find_nearest_model(self, target_teff, target_logg, target_feh, target_turb):
        """
        Nearest-neighbor selection in (Teff, logg, [Fe/H]) space.

        Preference:
        - Use only models with matching turbulence if any exist.
        - Otherwise, fall back to any turbulence.
        """
        if not self.available_models:
            return None, f"No models found in {self.config.model_atmosphere_path}"

        same_turb = [m for m in self.available_models if m["turb"] == str(target_turb)]
        candidates = same_turb if same_turb else list(self.available_models)
        turb_note = "" if same_turb else f" (no models with t={target_turb}; turbulence not matched)"

        # Scale factors roughly matching common MARCS grid spacings
        teff_scale = 250.0
        logg_scale = 0.5
        feh_scale = 0.5

        def dist2(m):
            dt = (m["teff"] - target_teff) / teff_scale
            dg = (m["logg"] - target_logg) / logg_scale
            dz = (m["feh"] - target_feh) / feh_scale
            return dt * dt + dg * dg + dz * dz

        best = min(candidates, key=dist2)
        msg = (
            f"Nearest neighbor selected{turb_note}: {best['filename']} "
            f"(Teff={best['teff']}, logg={best['logg']}, FeH={best['feh']}, t={best['turb']})"
        )
        return best, msg

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
    if isinstance(params, Mapping):
        request = _resolve_synthesis_request(params)
        teff = request["teff"]
        logg = request["logg"]
        feh = request["feh"]
        model_turb_str = request["model_turb_str"]
        synthesis_turb_raw = request["synthesis_turb_raw"]
        synthesis_abundances = request["abundances"]
        mode_value = params.get("mode")
    else:
        teff, logg, feh, model_turb_str = params
        teff = int(teff)
        logg = float(logg)
        feh = float(feh)
        model_turb_str = _normalize_turbulence_id(model_turb_str)
        synthesis_turb_raw = model_turb_str
        synthesis_abundances = {}
        mode_value = None

    # Map turb_str to float for babsma input if needed
    turb_val = _coerce_microturbulence_value(synthesis_turb_raw, fallback=model_turb_str)

    model_file = get_model_filename(teff, logg, feh, model_turb_str)
    model_path = os.path.join(config.model_atmosphere_path, model_file)

    output_mode = _normalize_output_mode(getattr(config, "output_mode", "Flux"))
    calculation_mode = "NLTE" if config.nlte else "LTE"
    base_name = get_synthesis_output_stem_from_params(
        {
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "t_value": model_turb_str,
            "turbvel": synthesis_turb_raw,
            "lam_min": config.lambda_min,
            "lam_max": config.lambda_max,
            "lam_step": config.lambda_step,
            "output_mode": output_mode,
            "calculation_mode": calculation_mode,
            "mode": mode_value,
            **synthesis_abundances,
        },
        default_output_mode=output_mode,
        default_calculation_mode=calculation_mode,
    )
    log_file = os.path.join(config.log_dir, f"{base_name}.log")
    opac_path = os.path.join(config.project_root, config.model_opac_dir, f"{base_name}.opac")
    is_intensity = output_mode == "Intensity"
    alpha_fe, r_proc, s_proc, individual_abundance_block = _build_abundance_controls(synthesis_abundances)
    individual_abundance_count = len(
        [line for line in individual_abundance_block.splitlines() if line.strip()]
    )
    nlte_ascii_info: Optional[Dict[str, Any]] = None

    def build_result(status: str, message: str, *, output_path: str = "", log_path: str = ""):
        extra_params: Dict[str, Any] = {}
        if nlte_ascii_info is not None:
            extra_params.update(
                {
                    "nlte_ascii_departure_file": nlte_ascii_info["departure_file"],
                    "nlte_ascii_target_abundance": f"{nlte_ascii_info['target_abundance']:+0.3f}",
                    "nlte_ascii_matched_abundance": f"{nlte_ascii_info['matched_abundance']:+0.3f}",
                    "nlte_ascii_species": nlte_ascii_info["species"],
                    "nlte_ascii_abundance_column": nlte_ascii_info["abundance_column"],
                }
            )
        return {
            "base_name": base_name,
            "params": {
                "teff": teff,
                "logg": logg,
                "feh": feh,
                "turb": model_turb_str,
                "turbvel": synthesis_turb_raw,
                "abundances": dict(synthesis_abundances),
                **extra_params,
            },
            "status": status,
            "message": message,
            "output_path": output_path,
            "log_path": log_path,
        }
    
    # Check if output exists and skip if force is False
    expected_outputs = []
    if is_intensity:
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
            return build_result(
                "skipped",
                "Output already exists",
                output_path=expected_outputs[0],
                log_path=log_file,
            )
    
    # If exact model doesn't exist, pick a nearest-neighbor MARCS atmosphere (no interpolation).
    model_input_path = model_path
    if not os.path.exists(model_path):
        model_lib = ModelInterpolator(config)
        nearest, message = model_lib.find_nearest_model(teff, logg, feh, model_turb_str)
        if not nearest:
            return build_result("error", f"Model missing and nearest-neighbor selection failed: {message}")
        model_input_path = nearest["path"]

    nlte_info_file_for_run = (
        os.path.abspath(str(config.nlte_info_file))
        if config.nlte_info_file
        else os.path.join(config.project_root, "DATA", "SPECIES_LTE_NLTE.dat")
    )
    if config.nlte and isinstance(params, Mapping):
        try:
            nlte_ascii_info = _resolve_nlte_ascii_runtime_info(
                params=params,
                config=config,
                model_input_path=model_input_path,
            )
        except Exception as exc:
            return build_result(
                "error",
                f"NLTE ASCII departure selection failed: {exc}",
                log_path=log_file,
            )
        if nlte_ascii_info is not None:
            nlte_info_file_for_run = str(nlte_ascii_info["nlte_info_file"])

    # Check if model is a standard MARCS model or interpolated
    is_marcs = True
    try:
        with open(model_input_path, 'r') as f:
            first_line = f.readline()
            if "INTERPOL" in first_line:
                is_marcs = False
    except:
        pass # Assume MARCS if read fails? Or fail later.

    marcs_flag = '.true.' if is_marcs else '.false.'

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Starting synthesis for {base_name}\n")
        if nlte_ascii_info is not None:
            log.write(
                "NLTE ASCII departure override: "
                f"species={nlte_ascii_info['species']} "
                f"column={nlte_ascii_info['abundance_column']} "
                f"target_abundance={nlte_ascii_info['target_abundance']:+0.3f} "
                f"matched_abundance={nlte_ascii_info['matched_abundance']:+0.3f} "
                f"model_stem={nlte_ascii_info['model_stem']} "
                f"file={nlte_ascii_info['departure_file']}\n"
            )
        
        # ---------------------------------------------------------------------
        # Step 1: BABSMA (Continuous Opacity)
        # ---------------------------------------------------------------------
        babsma_input = f"""'LAMBDA_MIN:'  '{config.lambda_min}'
'LAMBDA_MAX:'  '{config.lambda_max}'
'LAMBDA_STEP:' '{config.lambda_step}'
'MODELINPUT:' '{model_input_path}'
'MARCS-FILE:' '{marcs_flag}'
'MODELOPAC:' '{opac_path}'
'ABUND_SOURCE:' 'magg'
'METALLICITY:'    '{feh}'
'ALPHA/Fe   :'    '{alpha_fe}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{r_proc}'
'S-PROCESS  :'    '{s_proc}'
'INDIVIDUAL ABUNDANCES:'   '{individual_abundance_count}'
{individual_abundance_block}'XIFIX:' 'T'
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
                log.flush()
                return build_result(
                    "error",
                    _with_turbospectrum_log_context(
                        f"babsma failed (rc={process.returncode})",
                        log_file,
                    ),
                    output_path=expected_outputs[0],
                    log_path=log_file,
                )
        except Exception as e:
            log.flush()
            return build_result(
                "exception",
                _with_turbospectrum_log_context(f"babsma execution failed: {e}", log_file),
                log_path=log_file,
            )

        # ---------------------------------------------------------------------
        # Step 2: BSYN (Spectral Synthesis)
        # ---------------------------------------------------------------------
        
        # Determine synthesis mode.
        # For plane-parallel models, Turbospectrum outputs intensities for 12
        # standard mu angles in a single file, so we do not loop over angles.
        
        synthesis_runs = []
        if is_intensity:
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
'NLTEINFOFILE:'  '{nlte_info_file_for_run if config.nlte else (config.nlte_info_file if config.nlte_info_file else 'DATA/SPECIES_LTE_NLTE.dat')}'
'LAMBDA_MIN:'     '{config.lambda_min}'
'LAMBDA_MAX:'     '{config.lambda_max}'
'LAMBDA_STEP:'    '{config.lambda_step}'
'INTENSITY/FLUX:' '{mode_str}'
'MODELOPAC:' '{opac_path}'
'RESULTFILE :' '{current_result_file}'
'ABUND_SOURCE:'   'magg'
'METALLICITY:'    '{feh}'
'ALPHA/Fe   :'    '{alpha_fe}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{r_proc}'
'S-PROCESS  :'    '{s_proc}'
'INDIVIDUAL ABUNDANCES:'   '{individual_abundance_count}'
{individual_abundance_block}'ISOTOPES : ' '0'
'LIST_OF_LINELISTS:' '{config.linelist_file_path}'
'SPHERICAL:'  'F'
  30
  {float(config.resolution):.2f}
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
                    log.flush()
                    return build_result(
                        "error",
                        _with_turbospectrum_log_context(
                            f"bsyn failed ({mode_str}, rc={process.returncode})",
                            log_file,
                        ),
                        output_path=current_result_file,
                        log_path=log_file,
                    )
            except Exception as e:
                log.flush()
                return build_result(
                    "exception",
                    _with_turbospectrum_log_context(f"bsyn execution failed: {e}", log_file),
                    log_path=log_file,
                )

    return build_result(
        "success",
        "Synthesis complete",
        output_path=expected_outputs[0],
        log_path=log_file,
    )

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
    
    # Determine number of CPUs, respecting HPC scheduler assignments
    num_cpus = determine_worker_count(config)
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(run_single_synthesis, tasks)
        
    end_time = time.time()

    # Report results
    print("\n--- Summary ---")
    status_counts = Counter(res["status"] for res in results)
    summary_lines = []

    for res in results:
        params = res["params"]
        line = (
            f"{res['status'].upper():<9} {res['base_name']} "
            f"(Teff={params['teff']}, logg={params['logg']}, FeH={params['feh']}, turb={params['turb']}): "
            f"{res['message']}"
        )
        print(line)
        summary_lines.append(line)

    print(f"\nCompleted {status_counts.get('success', 0)}/{len(grid_points)} calculations in {end_time - start_time:.2f} seconds.")

    summary_header = [
        f"Turbospectrum synthesis summary - {datetime.now().isoformat()}",
        f"Total grid points: {len(grid_points)}",
        "Status counts: " + ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        ),
        "",
    ]

    summary_path = os.path.join(config.log_dir, f"synthesis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(summary_path, "w") as summary_file:
        summary_file.write("\n".join(summary_header + summary_lines))

    print(f"Summary log written to {summary_path}")

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
        cfg_data = _normalize_config_dict(cfg_data, default_project_root=project_root)
        accepted_fields = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
        cfg_data = {k: v for k, v in cfg_data.items() if k in accepted_fields}
        if 'project_root' not in cfg_data:
            cfg_data['project_root'] = project_root
        config = TurbospectrumConfig(**cfg_data)
    else:
        config = TurbospectrumConfig(project_root=project_root)
    
    # Apply force flag
    if force_flag:
        config.force = True
    
    # Load grid points from external file if specified
    if config.grid_points_file:
        print(f"Loading grid points from file: {config.grid_points_file}")
        grid_points = list(load_grid_points_from_csv(config.grid_points_file, config.project_root))
        print(f"Loaded {len(grid_points)} grid points from file")
    else:
        grid_points = config.grid_points
    
    # Example: enable intensity calculation
    # config.output_mode = "Intensity"
    # config.mu_angles = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    try:
        run_grid(config, grid_points)
    except LinelistValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

if __name__ == "__main__":
    main()
