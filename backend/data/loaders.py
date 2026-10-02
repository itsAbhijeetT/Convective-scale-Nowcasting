import numpy as np
import xarray as xr
import os
import glob

def dbz_to_rain_rate(dbz: np.ndarray) -> np.ndarray:
    """
    Convert radar reflectivity (dBZ) to rain rate (mm/hr) using the Marshall-Palmer relation.
    Z = 200 * R^1.6
    Z = 10^(dBZ/10)
    """
    z = 10.0 ** (dbz / 10.0)
    rain_rate = (z / 200.0) ** (1 / 1.6)
    # Filter out very low values to avoid noise
    rain_rate[rain_rate < 0.1] = 0.0
    return rain_rate

def rain_rate_to_dbz(rain_rate: np.ndarray) -> np.ndarray:
    """
    Convert rain rate (mm/hr) to radar reflectivity (dBZ).
    """
    z = 200.0 * (rain_rate ** 1.6)
    # Avoid log of 0
    z = np.clip(z, a_min=1e-5, a_max=None)
    dbz = 10.0 * np.log10(z)
    dbz[rain_rate < 0.1] = 0.0
    return dbz

def load_npy_sequence(directory: str) -> np.ndarray:
    """
    Load a sequence of NPY files from a directory.
    Returns array of shape (T, H, W)
    """
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        raise ValueError(f"No .npy files found in {directory}")
    
    frames = [np.load(f) for f in files]
    return np.stack(frames, axis=0)

def load_netcdf_sequence(file_path: str, variable: str = 'dbz') -> np.ndarray:
    """
    Load sequence from a single NetCDF file containing multiple time steps.
    """
    ds = xr.open_dataset(file_path)
    return ds[variable].values
