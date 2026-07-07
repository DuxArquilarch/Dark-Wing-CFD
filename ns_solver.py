# ================================================================= #
# ns_solver.py — Navier-Stokes 2D solver (incompressible, proj.)    #
# ================================================================= #

import math
import numpy as np
from matplotlib.path import Path as MplPath

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(fn):
        return fn

@njit
def _ns_momentum(vx, vy, p, rho, nu, dl, dt):
    """
    One substep: advection + diffusion + pressure gradient.
    Returns updated vx, vy, and max velocity magnitude.
    """
    nx, ny = vx.shape
    nvx = vx.copy()
    nvy = vy.copy()
    max_v = 1e-6

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Advection (upwind)
            u = vx[i, j]
            v = vy[i, j]

            if u >= 0:
                dvx_dx = (vx[i,   j] - vx[i-1, j]) / dl
                dvy_dx = (vy[i,   j] - vy[i-1, j]) / dl
            else:
                dvx_dx = (vx[i+1, j] - vx[i,   j]) / dl
                dvy_dx = (vy[i+1, j] - vy[i,   j]) / dl

            if v >= 0:
                dvx_dy = (vx[i, j  ] - vx[i, j-1]) / dl
                dvy_dy = (vy[i, j  ] - vy[i, j-1]) / dl
            else:
                dvx_dy = (vx[i, j+1] - vx[i, j  ]) / dl
                dvy_dy = (vy[i, j+1] - vy[i, j  ]) / dl

            # Viscous diffusion
            lap_vx = (vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1] - 4.0*vx[i,j]) / dl**2
            lap_vy = (vy[i+1,j] + vy[i-1,j] + vy[i,j+1] + vy[i,j-1] - 4.0*vy[i,j]) / dl**2

            # Pressure gradient
            dpdx = (p[i+1, j] - p[i-1, j]) / (2.0 * dl)
            dpdy = (p[i, j+1] - p[i, j-1]) / (2.0 * dl)

            nvx[i, j] = vx[i,j] + dt * (-u*dvx_dx - v*dvx_dy + nu*lap_vx - dpdx/rho)
            nvy[i, j] = vy[i,j] + dt * (-u*dvy_dx - v*dvy_dy + nu*lap_vy - dpdy/rho)

            vm = math.sqrt(nvx[i,j]**2 + nvy[i,j]**2)
            if vm > max_v:
                max_v = vm

    return nvx, nvy, max_v


@njit
def _ns_pressure(vx, vy, p, rho, dl, dt, n_poisson=20):
    """Pressure Poisson equation (Jacobi iterations)."""
    nx, ny = vx.shape
    np_arr = p.copy()
    for _ in range(n_poisson):
        pp = np_arr.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                div = ((vx[i+1,j] - vx[i-1,j]) + (vy[i,j+1] - vy[i,j-1])) / (2.0*dl)
                np_arr[i,j] = (pp[i+1,j] + pp[i-1,j] + pp[i,j+1] + pp[i,j-1]) / 4.0 \
                              - rho * dl**2 / (4.0 * dt) * div
    return np_arr


def ns_step(vx, vy, p, objet, rho, mu, dl, dt, v_inlet):
    """Full N-S projection step: momentum → pressure → projection → BC."""
    nu = mu / rho

    # Clamp dt for stability (CFL + diffusion)
    v_max_est = max(float(np.abs(vx).max()), float(np.abs(vy).max()), v_inlet, 1e-6)
    dt_cfl    = 0.35 * dl / v_max_est
    dt_diff   = 0.25 * dl**2 / (nu + 1e-12)
    dt        = min(dt, dt_cfl, dt_diff)

    vx, vy, v_max = _ns_momentum(vx, vy, p, rho, nu, dl, dt)
    p             = _ns_pressure(vx, vy, p, rho, dl, dt)

    # Velocity correction (projection)
    vx[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dl)
    vy[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dl)

    # No-slip on body
    vx[objet == 1] = 0.0
    vy[objet == 1] = 0.0

    # Inlet BC (left boundary)
    vx[0, :]  = v_inlet
    vy[0, :]  = 0.0

    # Outlet BC (right boundary) — zero gradient
    vx[-1, :] = vx[-2, :]
    vy[-1, :] = vy[-2, :]

    # Top/bottom — slip
    vx[:, 0]  = vx[:, 1]
    vx[:, -1] = vx[:, -2]
    vy[:, 0]  = 0.0
    vy[:, -1] = 0.0

    # Velocity clipping
    v_lim = v_inlet * 4.0
    np.clip(vx, -v_lim, v_lim, out=vx)
    np.clip(vy, -v_lim, v_lim, out=vy)

    return vx, vy, p, v_max, dt


def profile_to_mask(pts, aoa_deg, Nx, Ny):
    """Rasterise an airfoil profile into a boolean obstacle mask."""
    aoa  = np.radians(aoa_deg)
    c, s = math.cos(aoa), math.sin(aoa)
    rot  = np.array([[c, -s], [s, c]])

    pts_c       = pts.copy()
    pts_c[:, 0] -= 0.5
    pts_rot      = (rot @ pts_c.T).T

    bx_min, bx_max = pts_rot[:, 0].min(), pts_rot[:, 0].max()
    by_min, by_max = pts_rot[:, 1].min(), pts_rot[:, 1].max()
    bx_c = (bx_min + bx_max) / 2.0
    by_c = (by_min + by_max) / 2.0

    # Scale so chord spans ~55 % of the domain width; place at 40 % from inlet
    scale = Nx * 0.55
    pts_f = np.zeros_like(pts_rot)
    pts_f[:, 0] = (pts_rot[:, 0] - bx_c) * scale + Nx * 0.40
    pts_f[:, 1] = Ny / 2.0 - (pts_rot[:, 1] - by_c) * scale

    margin = 12
    x_out = max(0, -pts_f[:, 0].min() + margin, pts_f[:, 0].max() - (Nx - margin))
    y_out = max(0, -pts_f[:, 1].min() + margin, pts_f[:, 1].max() - (Ny - margin))
    if x_out > 0 or y_out > 0:
        shrink = min(
            (Nx - 2*margin) / ((bx_max - bx_min) * scale + 1e-9),
            (Ny - 2*margin) / ((by_max - by_min) * scale + 1e-9)
        )
        scale *= shrink
        pts_f[:, 0] = (pts_rot[:, 0] - bx_c) * scale + Nx * 0.40
        pts_f[:, 1] = Ny / 2.0 - (pts_rot[:, 1] - by_c) * scale

    # Rasterise with matplotlib Path (no cv2 dependency)
    poly_path = MplPath(pts_f)
    xs = np.arange(Nx); ys = np.arange(Ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    mask = poly_path.contains_points(grid_pts).reshape(Nx, Ny).astype(np.uint8)
    return mask


