# ================================================================= #
# panel_method.py — top-down panel-method potential flow            #
# ================================================================= #

import numpy as np

def panel_velocity_field(poly_x, poly_y, U_inf, X, Y):
    """
    Potential-flow panel method (source panels) + bound-vortex sheet
    representing the lifting wing.

    Bound vortex: elliptic spanload from lifting-line theory
        Gamma(y) = Gamma_0 * sqrt(1 - (y/b)^2)
        Gamma_0  = pi/2 * U_inf * b * CL_ref   (CL_ref ≈ 0.4)
    Trailing vortices: Helmholtz — each spanwise strip sheds two counter-
        rotating semi-infinite filaments at ±y_v, with strength dGamma/dy * dy.
    """
    n   = len(poly_x) - 1
    xm  = 0.5*(poly_x[:-1] + poly_x[1:])
    ym  = 0.5*(poly_y[:-1] + poly_y[1:])
    dx  = poly_x[1:] - poly_x[:-1]
    dy  = poly_y[1:] - poly_y[:-1]
    L   = np.sqrt(dx**2 + dy**2) + 1e-14
    tx  = dx / L;  ty = dy / L
    nx  = -ty;     ny = tx
    sigma = -2.0 * U_inf * nx

    Xf = X.ravel(); Yf = Y.ravel()
    U_ind = np.zeros(Xf.size)
    V_ind = np.zeros(Yf.size)

    # Source panels
    for j in range(n):
        dxf = Xf - xm[j]; dyf = Yf - ym[j]
        r2  = dxf**2 + dyf**2 + 1e-9
        fac = sigma[j] * L[j] / (2.0*np.pi)
        U_ind += fac * dxf / r2
        V_ind += fac * dyf / r2

    # Bound vortex sheet: elliptic spanwise loading
    half_span   = np.max(np.abs(poly_y))
    CL_ref      = 0.40
    Gamma_0     = 0.5 * np.pi * U_inf * half_span * CL_ref   # root circulation
    n_vort      = 40
    yv          = np.linspace(-half_span * 0.98, half_span * 0.98, n_vort)
    dy_v        = yv[1] - yv[0]
    x0_chord    = 0.25 * (np.max(poly_x) - np.min(poly_x)) + np.min(poly_x)  # quarter-chord

    for i, yvi in enumerate(yv):
        eta   = yvi / half_span
        Gamma = Gamma_0 * np.sqrt(max(1.0 - eta**2, 0.0))
        dGam  = Gamma_0 * (-eta / (np.sqrt(max(1.0 - eta**2, 1e-9)) * half_span))
        dg    = Gamma * dy_v / (half_span + 1e-9)

        dxf = Xf - x0_chord; dyf = Yf - yvi
        r2  = dxf**2 + dyf**2 + 1e-9
        # Bound segment: u = -dGam/(2pi) * dy/r^2, v = +dGam/(2pi)*dx/r^2
        U_ind += -dg / (2.0 * np.pi) * dyf / r2
        V_ind +=  dg / (2.0 * np.pi) * dxf / r2

    # Trailing vortex filaments: Helmholtz, semi-infinite from x0_chord to +inf
    # For a semi-infinite filament along +x from (x0, yv):
    #   u_theta = Gamma/(4pi*r) * (1 + cos(theta_end))
    # Approximated as a long finite set of segments along x.
    n_wake   = 30
    x_te     = np.max(poly_x)
    wake_xs  = np.linspace(x_te, x_te + 4.0, n_wake)
    for i, yvi in enumerate(yv):
        eta     = yvi / half_span
        dGam_dy = Gamma_0 * (-eta / (np.sqrt(max(1.0 - eta**2, 1e-9)) * half_span))
        d_gamma = dGam_dy * (yv[1] - yv[0]) if n_vort > 1 else 0.0
        for xw in wake_xs:
            dxf = Xf - xw; dyf = Yf - yvi
            r2  = dxf**2 + dyf**2 + 1e-9
            fac = d_gamma / (2.0 * np.pi * r2) * (4.0 / n_wake)
            U_ind += -fac * dyf
            V_ind +=  fac * dxf

    return (U_inf + U_ind).reshape(X.shape), V_ind.reshape(Y.shape)


