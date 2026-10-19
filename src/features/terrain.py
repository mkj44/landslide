"""Digital Elevation Model (DEM) terrain derivatives calculation."""

import numpy as np
from typing import Dict, Tuple


def compute_terrain_derivatives_2d(
    dem: np.ndarray,
    cell_size: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Calculates slope, aspect, profile curvature, planform curvature, and TRI from a 2D DEM array.

    Uses Horn (1981) and Zevenbergen & Thorne (1987) 3x3 finite-difference kernels.

    Args:
        dem: 2D numpy array of elevation in meters.
        cell_size: Spatial resolution of cells in meters (default 30m for SRTM/ALOS).

    Returns:
        Dictionary with keys 'slope_deg', 'aspect_deg', 'prof_curvature', 'plan_curvature', 'tri'.
    """
    # Pad borders with reflection to prevent edge artifacts
    padded = np.pad(dem, pad_width=1, mode="edge")

    # 3x3 window components
    z1 = padded[:-2, :-2]
    z2 = padded[:-2, 1:-1]
    z3 = padded[:-2, 2:]
    z4 = padded[1:-1, :-2]
    z5 = padded[1:-1, 1:-1] # Center
    z6 = padded[1:-1, 2:]
    z7 = padded[2:, :-2]
    z8 = padded[2:, 1:-1]
    z9 = padded[2:, 2:]

    # Partial derivatives
    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell_size)
    dz_dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell_size)

    # Slope in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect in degrees (0 to 360, compass heading)
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 360.0 - aspect_deg + 90.0) % 360.0

    # Second derivatives for curvature
    d2z_dx2 = ((z4 + z6) - 2 * z5) / (cell_size**2)
    d2z_dy2 = ((z2 + z8) - 2 * z5) / (cell_size**2)
    d2z_dxdy = ((z3 + z7) - (z1 + z9)) / (4 * (cell_size**2))

    p = dz_dx**2 + dz_dy**2 + 1e-8
    q = p + 1.0

    # Profile curvature (rate of slope change in gradient direction; positive = convex, negative = concave)
    prof_curv = -((dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy + dz_dy**2 * d2z_dy2) / (p * (q**1.5)))

    # Planform curvature (contour curvature perpendicular to gradient direction)
    plan_curv = -((dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy + dz_dx**2 * d2z_dy2) / (p**1.5))

    # Terrain Ruggedness Index (Riley et al. 1999): mean absolute elevation diff across neighbors
    neighbors = [z1, z2, z3, z4, z6, z7, z8, z9]
    tri = np.mean([np.abs(n - z5) for n in neighbors], axis=0)

    return {
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "prof_curvature": prof_curv,
        "plan_curvature": plan_curv,
        "tri": tri,
    }


def estimate_point_terrain(
    elevation_m: float,
    lat: float,
    lon: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimates physical terrain parameters for a geospatial point based on geographic location and elevation.

    In the Western Ghats mountain scarp, elevation strongly correlates with slope steepness,
    escarpment aspect (south-west monsoonal exposure), and ruggedness.
    """
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 100) + seed) % (2**32))

    # Slope model: Western Ghats scarp has average slope 22-38 deg at mid-high elevation
    base_slope = np.clip(12.0 + (elevation_m / 65.0) + rng.normal(0, 4.0), 3.0, 55.0)

    # Aspect: Western slopes (facing Arabian Sea monsoon: 220° to 290°) receive the highest moisture
    # Western Ghats orientation is roughly NNW-SSE (axis ~340°-160°)
    aspect = (245.0 + rng.normal(0, 35.0)) % 360.0

    # Curvature: positive = convex ridge, negative = concave hollow (debris flow accumulation zone)
    prof_curv = float(rng.normal(-0.02, 0.05))
    plan_curv = float(rng.normal(-0.015, 0.04))

    # Terrain Ruggedness Index (TRI) in meters
    tri = float(np.clip(base_slope * 1.8 + rng.normal(0, 3.0), 2.0, 120.0))

    return {
        "elevation_m": float(elevation_m),
        "slope_deg": float(base_slope),
        "aspect_deg": float(aspect),
        "prof_curvature": prof_curv,
        "plan_curvature": plan_curv,
        "tri": tri,
    }
