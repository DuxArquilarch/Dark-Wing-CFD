# ================================================================= #
# wing_presets.py — built-in wing planform shape generators         #
# ================================================================= #

import numpy as np

_imported_stl_planform = [None]   # [0] = (poly_x, poly_y) or None

def _stl_profile_shape():
    """Returns the imported-STL planform, or falls back to a tapered wing."""
    if _imported_stl_planform[0] is not None:
        return _imported_stl_planform[0]
    # No STL loaded yet — fall back so the selector never hard-crashes
    return _taper_wing()

def _close(x, y):
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.append(x, x[0]); y = np.append(y, y[0])
    return x, y

def _centre(x, y):
    cx = (x.max() + x.min()) / 2.0
    return x - cx, y

def _wing_from_halves(x_le, x_te, y_span, n):
    sx = np.concatenate([x_le, x_te[::-1]])
    sy = np.concatenate([y_span, y_span[::-1]])
    px = np.concatenate([x_te, x_le[::-1]])
    py = -np.concatenate([y_span, y_span[::-1]])
    xc = np.concatenate([sx, px]); yc = np.concatenate([sy, py])
    xc, yc = _centre(xc, yc); xc, yc = _close(xc, yc)
    return xc, yc

def _taper_wing(half_span=1.2, root_chord=0.55, tip_chord=0.14, sweep_le=27.0, n=90):
    y = np.linspace(0, half_span, n)
    x_le = y * np.tan(np.radians(sweep_le))
    chord = root_chord + (tip_chord - root_chord) * (y / half_span)
    return _wing_from_halves(x_le, x_le + chord, y, n)

def _delta_wing(half_span=1.0, root_chord=1.1, sweep_le=58.0, n=80):
    y = np.linspace(0, half_span, n)
    x_le = y * np.tan(np.radians(sweep_le))
    x_te = np.where(x_le < root_chord, root_chord, x_le)
    return _wing_from_halves(x_le, x_te, y, n)

def _elliptical_wing(half_span=1.2, root_chord=0.55, n=120):
    t = np.linspace(0, 1, n); y = half_span * t
    chord = root_chord * np.sqrt(np.maximum(1.0 - t**2, 0.0))
    return _wing_from_halves(-chord/2.0, chord/2.0, y, n)

def _arrow_wing(half_span=1.15, root_chord=0.75, tip_chord=0.16,
                le_sweep=42.0, te_sweep=5.0, n=80):
    y = np.linspace(0, half_span, n)
    x_le = y * np.tan(np.radians(le_sweep))
    x_te = np.maximum(root_chord - y * np.tan(np.radians(te_sweep)), x_le + 0.05)
    return _wing_from_halves(x_le, x_te, y, n)

def _flying_wing(half_span=1.4, root_chord=1.0, tip_chord=0.22, le_sweep=33.0, n=80):
    y = np.linspace(0, half_span, n)
    x_le = y * np.tan(np.radians(le_sweep))
    chord = root_chord + (tip_chord - root_chord) * (y / half_span)
    x_te  = x_le + chord
    inb   = y < half_span * 0.35
    x_te[inb] -= 0.18 * np.sin(np.pi * y[inb] / (half_span * 0.35))
    return _wing_from_halves(x_le, x_te, y, n)

def _rectangular_wing(half_span=1.0, chord=0.45, n=60):
    y = np.linspace(0, half_span, n)
    return _wing_from_halves(np.zeros(n), np.full(n, chord), y, n)

def _ballistic_wing(length=1.7, radius=0.17, nose_frac=0.35, tail_frac=0.18, n=140):
    """Bullet / ballistic-round planform: paraboloid ogive nose,
    cylindrical mid-body, tapered boat-tail base."""
    x = np.linspace(0.0, length, n)
    r = np.empty(n)

    nose_len  = nose_frac * length
    tail_len  = tail_frac * length
    body_end  = length - tail_len

    nose_mask = x <= nose_len
    r[nose_mask] = radius * np.sqrt(np.clip(x[nose_mask] / max(nose_len, 1e-9), 0.0, 1.0))

    body_mask = (x > nose_len) & (x <= body_end)
    r[body_mask] = radius

    tail_mask = x > body_end
    tt = (x[tail_mask] - body_end) / max(tail_len, 1e-9)
    r[tail_mask] = radius * (1.0 - 0.45 * tt)

    x_upper, y_upper = x, r
    x_lower, y_lower = x[::-1], -r[::-1]
    xc = np.concatenate([x_upper, x_lower])
    yc = np.concatenate([y_upper, y_lower])
    xc, yc = _centre(xc, yc)
    xc, yc = _close(xc, yc)
    return xc, yc

WING_SHAPES = {
    "Tapered Swept":   _taper_wing,
    "Delta":           _delta_wing,
    "Elliptical":      _elliptical_wing,
    "Arrow / Cranked": _arrow_wing,
    "Flying Wing":     _flying_wing,
    "Rectangular":     _rectangular_wing,
    "Ballistic":       _ballistic_wing,
    "Imported STL":    _stl_profile_shape,
}

