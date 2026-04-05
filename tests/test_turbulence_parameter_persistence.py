import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.merge_spectra_shards import _build_params_matrix as build_merged_params_matrix
from scripts.synthesize_spectra_from_zarr import _build_params_matrix as build_synth_params_matrix


def test_synthesis_params_preserve_turbvel_and_t_value() -> None:
    columns = {
        "teff": np.asarray([5000, 5250], dtype=np.int64),
        "logg": np.asarray([4.0, 4.5], dtype=np.float64),
        "feh": np.asarray([-0.5, 0.0], dtype=np.float64),
        "turbvel": np.asarray(["01", "02"], dtype=object),
        "t_value": np.asarray(["01", "03"], dtype=object),
        "a": np.asarray(["+0.00", "+0.10"], dtype=object),
    }

    params, param_names = build_synth_params_matrix(columns)

    assert param_names.tolist() == ["teff", "logg", "feh", "vmicro", "turbvel", "t_value", "a"]
    np.testing.assert_allclose(params[:, 3], np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(params[:, 4], np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(params[:, 5], np.asarray([1.0, 3.0], dtype=np.float32))


def test_merged_params_preserve_turbvel_and_t_value() -> None:
    columns = {
        "teff": np.asarray([5000], dtype=np.int64),
        "logg": np.asarray([4.0], dtype=np.float64),
        "feh": np.asarray([-0.5], dtype=np.float64),
        "turbvel": np.asarray(["02"], dtype=object),
        "t_value": np.asarray(["05"], dtype=object),
        "c": np.asarray(["+0.00"], dtype=object),
    }

    params, param_names = build_merged_params_matrix(columns)

    assert param_names.tolist() == ["teff", "logg", "feh", "vmicro", "turbvel", "t_value", "c"]
    np.testing.assert_allclose(params[:, 3], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(params[:, 4], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(params[:, 5], np.asarray([5.0], dtype=np.float32))
