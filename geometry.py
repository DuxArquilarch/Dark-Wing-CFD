# ================================================================= #
# geometry.py — STL/STEP loaders, profile extraction, NACA, presets #
# ================================================================= #

import os
import struct
import numpy as np

def parse_stl(filepath):
    """Read binary or ASCII STL → (N,3) float64 vertices."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
        is_ascii = False
        try:
            if header.decode('ascii', errors='ignore').strip().lower().startswith('solid'):
                f.seek(0)
                content = f.read().decode('ascii', errors='ignore')
                if 'facet normal' in content[:2000] and 'endsolid' in content[-1000:].lower():
                    is_ascii = True
        except Exception:
            pass
        f.seek(0)
        if is_ascii:
            verts = []
            for line in f:
                line = line.decode('ascii', errors='ignore')
                if 'vertex' in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except Exception:
                            continue
            arr = np.array(verts, dtype=np.float64)
            # Reconstruct per-triangle arrays (groups of 3)
            n = (len(arr) // 3) * 3
            arr = arr[:n]
            return arr[0::3], arr[1::3], arr[2::3]
        else:
            f.seek(80)
            num_tri = struct.unpack('<I', f.read(4))[0]
            dtype = np.dtype([
                ('normal', np.float32, 3), ('v0', np.float32, 3),
                ('v1', np.float32, 3),     ('v2', np.float32, 3),
                ('attr', np.uint16, 1)
            ])
            data = np.fromfile(f, dtype=dtype, count=num_tri)
            return (data['v0'].astype(np.float64),
                    data['v1'].astype(np.float64),
                    data['v2'].astype(np.float64))


def parse_step(filepath):
    """
    Extract 3D vertex cloud from STEP/STP (ISO 10303-21).
    Reads CARTESIAN_POINT entries without external libs.
    Returns (N,3) float64.
    """
    verts = []
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if 'CARTESIAN_POINT' not in line.upper():
                continue
            # e.g. #10=CARTESIAN_POINT('',(1.0,2.0,3.0));
            try:
                paren = line.index(',(') 
                inner = line[paren+2 : line.rindex(')')]
                coords = [float(c.strip()) for c in inner.split(',')]
                if len(coords) == 3:
                    verts.append(coords)
                elif len(coords) == 2:
                    verts.append([coords[0], coords[1], 0.0])
            except Exception:
                continue
    if len(verts) == 0:
        raise ValueError("No CARTESIAN_POINT data found in STEP file.")
    return np.array(verts, dtype=np.float64)


def _densify_triangles(v0, v1, v2, n_edge=12):
    """
    Sample n_edge points along each edge of every triangle.
    Returns (N,3) float64 — a dense point cloud even for low-poly meshes.
    """
    t = np.linspace(0, 1, n_edge)
    parts = []
    for ti in t:
        parts.append(v0 + ti * (v1 - v0))
        parts.append(v1 + ti * (v2 - v1))
        parts.append(v2 + ti * (v0 - v2))
    return np.vstack(parts)


def extract_2d_profile(tri_or_verts, z_tol=1e-3, n_bins=300):
    """
    Extract a closed XY airfoil profile from STL triangle data or a plain vertex cloud.

    Accepts either:
      - a tuple (v0, v1, v2) of per-triangle float64 arrays  [preferred]
      - a plain (N,3) vertex array                            [legacy / STEP]

    Strategy:
      1. Densify by sampling triangle edges (fixes sparse/low-poly STLs).
      2. Try projecting onto all three axis-pairs (XZ, XY, YZ) and pick
         the pair that produces the most bins with data.
      3. Fall back to the two highest-variance axes.
    """
    # --- Unpack input ---
    if isinstance(tri_or_verts, tuple):
        v0, v1, v2 = tri_or_verts
        dense = _densify_triangles(v0, v1, v2, n_edge=12)
    else:
        dense = np.asarray(tri_or_verts, dtype=np.float64)

    def _bin_profile(pts2d, n_bins):
        """Bin pts2d[:,0] → envelope of pts2d[:,1]. Returns (upper, lower) or None."""
        pts = np.unique(pts2d.round(6), axis=0)
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        if x_max - x_min < 1e-9:
            return None
        bins = np.linspace(x_min, x_max, n_bins)
        upper, lower = [], []
        for i in range(len(bins) - 1):
            mb = (pts[:, 0] >= bins[i]) & (pts[:, 0] < bins[i + 1])
            if np.any(mb):
                upper.append([bins[i], pts[mb, 1].max()])
                lower.append([bins[i], pts[mb, 1].min()])
        if len(upper) < 5:
            return None
        return np.array(upper), np.array(lower)

    def _build_profile(upper, lower):
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])[::-1]]
        profile = np.vstack([upper, lower])
        if not np.allclose(profile[0], profile[-1]):
            profile = np.vstack([profile, profile[0]])
        return profile

    # Try all three 2-D projections; pick the one with the most bins
    axis_pairs = [(0, 2), (0, 1), (1, 2)]   # XZ, XY, YZ
    best = None
    best_count = 0
    for a, b in axis_pairs:
        pts2d = dense[:, [a, b]]
        result = _bin_profile(pts2d, n_bins)
        if result is not None and len(result[0]) > best_count:
            best = result
            best_count = len(result[0])

    if best is not None:
        return _build_profile(*best)

    # Last resort: project onto the two highest-variance axes
    vars_ = np.var(dense, axis=0)
    order = np.argsort(vars_)[::-1]
    pts2d = dense[:, order[:2]]
    result = _bin_profile(pts2d, n_bins)
    if result is not None:
        return _build_profile(*result)

    raise ValueError("Could not extract 2D profile — check geometry orientation.")


def smooth_profile(pts, window=5):
    """
    Smooth a closed 2-D profile using a moving-average filter.

    Returns an array with exactly the same shape as the input,
    avoiding NumPy broadcasting errors.
    """
    pts = np.asarray(pts, dtype=np.float64)

    if len(pts) < window * 2:
        return pts.copy()

    smoothed = pts.copy()

    kernel = np.ones(window, dtype=np.float64) / float(window)

    for i in range(2):
        smoothed[:, i] = np.convolve(
            pts[:, i],
            kernel,
            mode='same'
        )

    # Preserve endpoints
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]

    return smoothed

def normalize_profile(pts):
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    chord = max(x_max - x_min, 1e-9)
    pts[:, 0] = (pts[:, 0] - x_min) / chord
    pts[:, 1] =  pts[:, 1] / chord
    return pts


def _naca4(m_d, p_d, t_d, n=100):
    """
    Analytic NACA 4-digit airfoil (m/100, p/10, t/100 as fractions).
    Returns closed (2n+1, 2) array in Selig ordering: TE→upper→LE→lower→TE.
    """
    m = m_d / 100.0   # max camber
    p = p_d / 10.0    # max camber position
    t = t_d / 100.0   # thickness fraction

    # Cosine spacing for sharper LE/TE resolution
    beta = np.linspace(0, np.pi, n + 1)
    xc   = 0.5 * (1.0 - np.cos(beta))

    # Thickness distribution (NACA 4-series standard coefficients)
    yt = (t / 0.2) * (0.2969*np.sqrt(xc)
                      - 0.1260*xc
                      - 0.3516*xc**2
                      + 0.2843*xc**3
                      - 0.1015*xc**4)

    # Camber line & gradient
    yc    = np.where(xc < p,
                     m/p**2 * (2*p*xc - xc**2),
                     m/(1-p)**2 * ((1-2*p) + 2*p*xc - xc**2))
    dyc   = np.where(xc < p,
                     2*m/p**2 * (p - xc),
                     2*m/(1-p)**2 * (p - xc))
    theta = np.arctan(dyc)

    xu = xc  - yt*np.sin(theta)
    yu = yc  + yt*np.cos(theta)
    xl = xc  + yt*np.sin(theta)
    yl = yc  - yt*np.cos(theta)

    # Selig ordering: TE → upper (reversed) → LE → lower → TE
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x, y]).astype(np.float64)


# Default NACA 2412 profile used as fallback
_NACA2412 = _naca4(2, 4, 12)


def load_geometry(filepath):
    """Load airfoil/wing geometry from .dat/.stl/.step/.stp file.
    Returns (pts, name) where pts is a normalized (N,2) float64 array."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    ext  = os.path.splitext(filepath)[1].lower()

    # ── STL ──────────────────────────────────────────────────────────
    if ext in ('.stl', '.stlb'):
        verts = parse_stl(filepath)
        pts   = extract_2d_profile(verts)
        pts   = smooth_profile(pts, window=5)
        return normalize_profile(pts), name

    # ── STEP / STP ───────────────────────────────────────────────────
    if ext in ('.step', '.stp'):
        verts = parse_step(filepath)
        pts   = extract_2d_profile(verts)
        pts   = smooth_profile(pts, window=5)
        return normalize_profile(pts), name

    # ── DAT / TXT ────────────────────────────────────────────────────
    try:
        raw = open(filepath, 'r', errors='ignore').readlines()
    except Exception:
        return _NACA2412.copy(), "NACA2412 (fallback)"

    def is_lednicer(lines):
        for line in lines[1:4]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                    if a > 2.0 and b > 2.0:
                        return True
                except Exception:
                    pass
        return False

    coords = []
    if is_lednicer(raw):
        skip = 0
        for i, line in enumerate(raw):
            parts = line.split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                    if a > 2.0 and b > 2.0 and i < 5:
                        skip = i + 1
                        break
                except Exception:
                    pass
        for line in raw[skip:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except Exception:
                    continue
    else:
        started = False
        for i, line in enumerate(raw):
            if i == 0 and not line.strip()[0:1].replace('-','').replace('.','').isdigit():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    if not started and abs(x) > 2.0:
                        continue
                    coords.append([x, y])
                    started = True
                except Exception:
                    continue

    if len(coords) < 6:
        return _NACA2412.copy(), "NACA2412 (fallback)"

    pts = np.array(coords, dtype=np.float64)
    return normalize_profile(pts), name


