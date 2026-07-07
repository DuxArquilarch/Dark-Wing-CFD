# ================================================================= #
# flow_field.py — FlowField solver class + aero coefficient stats   #
# ================================================================= #

import numpy as np
import scipy.ndimage as ndimage
from matplotlib.path import Path as MplPath

from panel_method import panel_velocity_field
from ns_solver import _ns_momentum, _ns_pressure

class FlowField:
    def __init__(self, nx=140, ny=110):
        self.NX, self.NY = nx, ny
        self.xi = np.linspace(-1.6, 2.6, nx)
        self.yi = np.linspace(-2.2, 2.2, ny)
        self.XX, self.YY = np.meshgrid(self.xi, self.yi, indexing='ij')
        self.U    = np.ones((nx, ny))
        self.V    = np.zeros((nx, ny))
        self.P    = np.ones((nx, ny))
        self.rho  = np.ones((nx, ny))
        self._mask = np.zeros((nx, ny), dtype=bool)

    def _build_mask(self, px, py):
        poly = MplPath(np.column_stack([px, py]))
        pts  = np.column_stack([self.XX.ravel(), self.YY.ravel()])
        self._mask = poly.contains_points(pts).reshape(self.NX, self.NY)

    def solve(self, mach, gamma, rho0, poly_x, poly_y,
              ns_iter=350, dt_init=1e-3):
        """
        Hybrid analytic + N-S planform solver.

        Physics layers (all additive on the grid):
          1. Panel-method potential flow — attached flow around body
          2. Trailing-edge Kármán vortex street — alternating shed vortex blobs
             that convect downstream and create the visible wake street
          3. Wingtip trailing vortices — counter-rotating Helmholtz vortex pair
             from each tip, spiralling inward and downstream
          4. Drag wake — low-velocity, high-pressure recirculation zone
             immediately behind the blunt/separated trailing region
          5. N-S viscous correction — short time-march that lets all the above
             interact, diffuse, and develop natural instabilities
        """
        U_inf = float(mach)
        p0    = 1.0 / gamma

        # ── Geometry ─────────────────────────────────────────────────
        px, py = poly_x.copy(), poly_y.copy()
        if not (np.isclose(px[0], px[-1]) and np.isclose(py[0], py[-1])):
            px = np.append(px, px[0]); py = np.append(py, py[0])
        self._build_mask(px, py)

        NX, NY   = self.NX, self.NY
        XX, YY   = self.XX, self.YY
        mask     = self._mask
        dl       = (self.xi[-1] - self.xi[0]) / NX

        # Wing geometry measurements
        half_span = float(np.max(np.abs(py)))
        x_le      = float(np.min(px))
        x_te      = float(np.max(px))
        chord     = x_te - x_le
        x_mid     = 0.5 * (x_le + x_te)

        # ── 1. Panel-method base flow ────────────────────────────────
        U_base, V_base = panel_velocity_field(px, py, U_inf, XX, YY)

        # ── 2. Trailing-edge Kármán vortex street ────────────────────
        # Physical basis: each shed vortex carries circulation
        #   Gamma_blob ~ St * U_inf * h   (h = wake half-width, St≈0.2)
        # blobs alternate sign (Kármán), offset ±h/2 laterally,
        # spaced lambda = U_inf/f = U_inf/(St*U_inf/h) = h/St chordwise.
        St          = 0.20
        wake_h      = half_span * 0.13          # wake half-width at TE
        lambda_s    = wake_h / St               # vortex street wavelength
        n_blobs     = 24
        blob_x      = np.linspace(x_te + 0.5*lambda_s,
                                  x_te + n_blobs * lambda_s, n_blobs)
        blob_delta  = wake_h * 0.5             # lateral offset (± half wake h)
        # Rankine core radius ~ blob spacing / 4
        blob_r2     = (lambda_s * 0.25) ** 2
        # Circulation per blob: Gamma = U_inf * lambda_s * St (Kármán)
        Gamma_blob  = U_inf * lambda_s * St * (1.0 + 0.3 * mach)

        U_wake = np.zeros_like(XX)
        V_wake = np.zeros_like(YY)
        Xf, Yf = XX.ravel(), YY.ravel()

        for k, bx in enumerate(blob_x):
            sign = 1.0 if k % 2 == 0 else -1.0
            for y_off, g_sign in [(+blob_delta, sign), (-blob_delta, -sign)]:
                dx = Xf - bx
                dy = Yf - y_off
                r2 = dx**2 + dy**2 + blob_r2
                fac = Gamma_blob / (2.0 * np.pi * r2)
                U_wake += (-g_sign * dy * fac).reshape(NX, NY)
                V_wake += ( g_sign * dx * fac).reshape(NX, NY)

        # ── 3. Wingtip trailing vortex pair (Biot-Savart, Rankine core) ─
        # Lifting-line theory: for elliptic loading,
        #   Gamma_root = pi/4 * U_inf * b * CL_mean
        # We estimate CL_mean ~ 0.4 (cruise-ish) so:
        #   Gamma_tip  = Gamma_root * 0.55   (roll-up concentration factor)
        # Vortex contracts inward: y_tip(x) = half_span*(1 - k*(x-x_te)/b)
        # with k ~ 0.12 (experimental roll-up rate).
        # Biot-Savart for a segment of a straight trailing vortex filament
        # lying along x, evaluated at (X,Y):
        #   u_theta = Gamma/(2*pi*r) * (1 - exp(-r^2/rc^2))   (Lamb-Oseen)
        #   components: du = -Gamma/(2pi) * dy/r^2_lo,  dv = +Gamma/(2pi) * dx/r^2_lo
        # The sign for the PORT tip vortex is OPPOSITE (Helmholtz).

        CL_mean    = 0.40
        Gamma_tip  = 0.55 * np.pi / 4.0 * U_inf * half_span * CL_mean
        n_tip_seg  = 60
        tip_xs     = np.linspace(x_te, x_te + 3.5, n_tip_seg)
        k_contrac  = 0.10                           # inward drift rate
        rc_tip     = max(half_span * 0.04, 0.01)   # Rankine core radius
        rc2_tip    = rc_tip ** 2

        U_tip = np.zeros_like(XX)
        V_tip = np.zeros_like(YY)

        for i, tx in enumerate(tip_xs):
            # Vortex core contracts toward centreline
            y_drift = k_contrac * (tx - x_te)
            segs = [
                # (y_centre, circulation_sign)
                # Starboard tip (+y): sheds clockwise vortex → V<0 inboard
                (+half_span - y_drift, +1.0),
                # Port tip (−y):      sheds counter-clockwise → V>0 inboard
                (-half_span + y_drift, -1.0),
            ]
            seg_gamma = Gamma_tip / n_tip_seg

            for y_vort, sign in segs:
                dx_v = Xf - tx
                dy_v = Yf - y_vort
                r2   = dx_v**2 + dy_v**2
                # Lamb-Oseen desingularisation: 1 - exp(-r^2/rc^2)
                lo   = 1.0 - np.exp(-r2 / (rc2_tip + 1e-30))
                fac  = sign * seg_gamma * lo / (2.0 * np.pi * (r2 + rc2_tip))
                # Biot-Savart: u_ind = -Gamma/(2pi) * dy/r^2, v_ind = +Gamma/(2pi)*dx/r^2
                U_tip += (-fac * dy_v).reshape(NX, NY)
                V_tip += ( fac * dx_v).reshape(NX, NY)

        # ── 4. Drag wake — elongated narrow tube + shear layers ──────
        #
        # Target: tight dark cylinder stretching far downstream with
        # bright shear-layer edges (matching reference wedge video).
        #   a) Wake width at TE ~13% half-span (narrow)
        #   b) Decay length = 4x chord (long persistence)
        #   c) Wake spreads slowly downstream (sqrt growth)
        #   d) Near-TE recirculation bubble explicit and strong
        #   e) Shear-layer velocity jets on wake edges -> bright halo
        #
        x_wake_ref  = x_te + 0.005
        # For slender bodies of revolution (e.g. the Ballistic preset) half_span
        # is just the body radius, which collapses the wake/recirculation bubble
        # to an invisible sliver. Floor the wake width to a fraction of chord so
        # the drag bubble stays visible for slender shapes too.
        wake_w0     = max(half_span * 0.13, chord * 0.03)
        decay_len   = chord * 4.0
        dist_x      = np.maximum(XX - x_wake_ref, 0.0)
        wake_y_spread = wake_w0 * (1.0 + np.sqrt(np.clip(dist_x / decay_len, 0, 4)))

        deficit = (U_inf * 0.75
                   * np.exp(-dist_x / decay_len)
                   * np.exp(-0.5 * (YY / (wake_y_spread + 1e-9)) ** 2))
        deficit = np.where(XX > x_wake_ref, deficit, 0.0)

        # Near-TE recirculation bubble: strong reverse flow right behind TE
        bubble_len = chord * 0.65
        bubble_w   = wake_w0 * 0.55
        recirc_x   = np.maximum(XX - x_te, 0.0)
        recirc_bubble = (U_inf * 0.55
                         * np.exp(-recirc_x / (chord * 0.20))
                         * np.exp(-0.5 * (YY / (bubble_w + 1e-9)) ** 2))
        recirc_bubble = np.where((XX > x_te) & (XX < x_te + bubble_len),
                                 recirc_bubble, 0.0)

        # Shear-layer velocity boost on wake edges -> bright halo
        shear_offset = wake_y_spread * 1.05
        shear_w      = wake_w0 * 0.18
        shear_boost  = (U_inf * 0.35
                        * np.exp(-dist_x / (decay_len * 1.5))
                        * (np.exp(-0.5 * ((YY - shear_offset) / (shear_w + 1e-9))**2)
                         + np.exp(-0.5 * ((YY + shear_offset) / (shear_w + 1e-9))**2)))
        shear_boost = np.where(XX > x_wake_ref, shear_boost, 0.0)

        # ── Compose initial velocity field ───────────────────────────
        vx = (U_base + U_wake + U_tip
              - deficit
              - recirc_bubble
              + shear_boost)
        vy = V_base + V_wake + V_tip

        # Enforce body no-slip
        vx[mask] = 0.0;  vy[mask] = 0.0

        # Pressure: Bernoulli from velocity magnitude
        V2 = vx**2 + vy**2
        p  = np.clip(p0 * (1.0 - 0.5 * (V2 - U_inf**2) / (U_inf**2 + 1e-12)),
                     p0 * 0.3, p0 * 2.5)
        p[mask] = p0

        # ── 5. Short N-S time-march — lets flow interact & develop ───
        Re_eff = max(150.0, 600.0 / (mach + 0.1))
        nu     = U_inf * (self.xi[-1] - self.xi[0]) / Re_eff
        objet  = mask.astype(np.uint8)

        # Asymmetric seed so wake can develop natural instability
        rng = np.random.default_rng(7)
        wake_region = (XX > x_te) & (np.abs(YY) < wake_w0 * 2.0)
        vy += np.where(wake_region,
                       rng.standard_normal((NX, NY)) * U_inf * 0.04, 0.0)

        dt = dt_init
        for step in range(ns_iter):
            vmax = max(float(np.abs(vx).max()), float(np.abs(vy).max()), U_inf, 1e-9)
            dt   = min(dt, 0.35 * dl / vmax, 0.25 * dl**2 / (nu + 1e-12))

            vx, vy, _ = _ns_momentum(vx, vy, p, rho0, nu, dl, dt)
            p          = _ns_pressure(vx, vy, p, rho0, dl, dt, n_poisson=20)
            vx[1:-1,1:-1] -= dt/rho0 * (p[2:,1:-1] - p[:-2,1:-1]) / (2.0*dl)
            vy[1:-1,1:-1] -= dt/rho0 * (p[1:-1,2:] - p[1:-1,:-2]) / (2.0*dl)

            vx[objet==1] = 0.0;  vy[objet==1] = 0.0;  p[objet==1] = p0
            vx[0,:]  = U_inf;    vy[0,:] = 0.0
            vx[-1,:] = vx[-2,:]; vy[-1,:] = vy[-2,:]
            vx[:,0]  = vx[:,1];  vx[:,-1] = vx[:,-2]
            vy[:,0]  = 0.0;      vy[:,-1] = 0.0

            np.clip(vx, -U_inf*5, U_inf*5, out=vx)
            np.clip(vy, -U_inf*5, U_inf*5, out=vy)

        # ── Compressibility: isentropic Euler relations ───────────────
        # For the planform Euler field we work in normalised units where
        # the free-stream speed equals `mach` (= U_inf) and the reference
        # speed of sound a0 = 1.  Local Mach is therefore |V|/a_local.
        #
        # Isentropic chain:
        #   T_loc/T0  = 1 - (gamma-1)/2 * (|V|/a0)^2 / (1 + (gamma-1)/2*M_inf^2)
        #   But in normalised units a0^2 = gamma*p0/rho0 so we use:
        #   Cp_incomp = 1 - V^2 / U_inf^2
        #   Cp_PG     = Cp_incomp / beta_inf          (Prandtl-Glauert, subsonic)
        #   For near/super-sonic use Karman-Tsien or full isentropic.
        m_eff  = float(np.clip(mach, 0.01, 5.00))
        M2_inf = m_eff ** 2
        gm1    = gamma - 1.0

        V2 = vx**2 + vy**2

        if m_eff < 0.98:
            # Prandtl-Glauert: Cp corrected, then isentropic p & rho
            beta_inf = np.sqrt(max(1.0 - M2_inf, 1e-6))
            Cp_inc   = 1.0 - V2 / (U_inf**2 + 1e-12)
            Cp_pg    = Cp_inc / beta_inf
            # Isentropic local pressure from Cp definition:
            # p_loc = p_inf + Cp * q_inf = p0 + Cp_pg * 0.5*rho0*U_inf^2
            q_inf = 0.5 * rho0 * U_inf**2
            ploc  = np.clip(p0 + Cp_pg * q_inf, 1e-3, 10.0)
        else:
            # Isentropic relations (works for M > 1 too):
            # T_loc/T0 = (a_loc/a0)^2 = 1 - (gamma-1)/2 * (V^2 - U_inf^2) / a0^2
            # a0^2 = gamma * p0 / rho0
            a0_sq = gamma * p0 / (rho0 + 1e-12)
            # Stagnation enthalpy: h0 = a0^2/(gamma-1) + U_inf^2/2
            h0    = a0_sq / gm1 + 0.5 * U_inf**2
            # local a^2 = (gamma-1)*(h0 - V^2/2)
            a_loc_sq = np.clip(gm1 * (h0 - 0.5 * V2), 1e-6, None)
            T_ratio  = a_loc_sq / a0_sq
            ploc     = np.clip(p0 * T_ratio ** (gamma / gm1), 1e-3, 10.0)

        rholoc = rho0 * (ploc / p0) ** (1.0 / gamma)

        # Local speed of sound and local Mach (used by scalar("mach"))
        a0_sq_ref  = gamma * p0 / (rho0 + 1e-12)
        a_loc_sq_f = np.clip(gm1 * (a0_sq_ref / gm1 + 0.5 * U_inf**2 - 0.5 * V2),
                             1e-6, None)
        self._a_loc  = np.sqrt(a_loc_sq_f)          # local speed of sound grid

        vx[mask] = 0.0; vy[mask] = 0.0
        ploc[mask] = p0; rholoc[mask] = rho0

        # Reduced smoothing sigma (0.25 vs 0.4) to keep wake edges sharp
        out = ~mask
        def _sm(f, s=0.25):
            return np.where(out, ndimage.gaussian_filter(f, s), f)

        self.U   = _sm(vx)
        self.V   = _sm(vy)
        self.P   = _sm(ploc)
        self.rho = rholoc

    def vmag(self):
        return np.sqrt(self.U**2 + self.V**2)

    def local_mach(self):
        """True local Mach = |V| / a_local (isentropic a from solve)."""
        a = getattr(self, '_a_loc', None)
        if a is None:
            return self.vmag()
        return self.vmag() / np.maximum(a, 1e-9)

    def temperature(self):
        return self.P / np.maximum(self.rho, 1e-6)

    def vorticity(self):
        # dV/dx - dU/dy with correct per-axis grid spacing
        dVdx = np.gradient(self.V, self.xi, axis=0)
        dUdy = np.gradient(self.U, self.yi, axis=1)
        return dVdx - dUdy

    def scalar(self, name):
        return {
            "pressure":    lambda: self.P,
            "mach":        self.local_mach,
            "vorticity":   self.vorticity,
            "temperature": self.temperature,
        }[name]()


def _smoothstep(x, lo, hi):
    """Cubic smoothstep, clamped to [0,1] outside [lo,hi]."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def aero_stats(mach, gamma, rho0, tc=0.12, CL=0.40, kappa=0.87):
    """
    Aerodynamic statistics with physically correct formulae.

    Mach cone (Mach angle):
        mu = arcsin(1/M)  — valid for M > 1, undefined below.

    Cp_min (isentropic, lowest stagnation-to-local pressure ratio):
        Cp_min = 2/(gamma*M^2) * [(1 + (gamma-1)/2*M^2*(1-V_max/a)^2)^(gamma/(gamma-1)) - 1]
        Simplified with V_max/a0 ≈ 1.2 M (empirical peak for thin wings):
        Computed from full isentropic chain.

    Wave / compressibility drag — three-regime model driven by the
    airfoil's REAL thickness ratio `tc` (max thickness / chord), instead
    of a fixed 0.12:

      1. Subcritical (M < M_dd): no wave drag. The critical/drag-divergence
         Mach number is estimated from Korn's equation:
             M_dd = kappa - tc - CL/10
         (kappa ≈ 0.87-0.89 for conventional airfoils, ~0.95 for
         supercritical sections — a standard conceptual-design relation,
         e.g. Raymer, "Aircraft Design: A Conceptual Approach".)

      2. Transonic drag-divergence rise (M_dd <= M < ~1.2): the familiar
         "sound-barrier" bump, using Raymer's empirical quartic fit:
             delta_CD_wave = 20 * (M - M_dd)^4
         This blows up rapidly as local shocks strengthen and is what
         produces the classic steep CD spike just below Mach 1.

      3. Supersonic linear theory (M >~ 1.2): Ackeret 2-D thin-airfoil
         result, now using the real thickness ratio:
             CD_wave = 4*(t/c)^2 / sqrt(M^2 - 1)

      A smooth (cubic) blend is applied over M in [1.0, 1.2] so the two
      asymptotic theories join continuously rather than jumping.

    Reynolds number with ISA sea-level values.
    """
    M    = float(mach)
    gm1  = gamma - 1.0
    beta = max(1e-6, np.sqrt(abs(1.0 - M**2)))

    # Pressure ratio at stagnation: p0/p_inf = (1 + gm1/2 * M^2)^(g/gm1)
    stag_ratio = (1.0 + 0.5 * gm1 * M**2) ** (gamma / gm1)

    # Cp_min: isentropic with V_max ≈ 1.2*U_inf (thin wing suction peak)
    V_ratio_sq = (1.2 * M) ** 2    # (V_max/a0)^2 in normalised units
    T_ratio_min = max(1.0 - 0.5 * gm1 * V_ratio_sq / (1.0 + 0.5 * gm1 * M**2), 1e-6)
    p_ratio_min = T_ratio_min ** (gamma / gm1)
    q_inf       = 0.5 * gamma * M**2   # non-dim dynamic pressure (p_inf=1/gamma)
    Cpmin       = (p_ratio_min - 1.0) / (q_inf + 1e-12)

    # ── Drag-divergence Mach number (Korn's equation) ───────────────────
    tc   = max(1e-3, float(tc))
    M_dd = kappa - tc - CL / 10.0
    M_dd = float(np.clip(M_dd, 0.30, 0.99))   # keep in a physically sane band

    # ── Branch 1/2: subsonic → transonic drag-divergence rise ──────────
    CD_dd = 20.0 * max(M - M_dd, 0.0) ** 4

    # ── Branch 3: Ackeret supersonic linear theory (real t/c) ───────────
    CD_ackeret = (4.0 * tc**2 / beta) if M > 1.0 else 0.0

    # ── Blend the two theories smoothly over M in [1.0, 1.2] ────────────
    w = _smoothstep(M, 1.0, 1.2)
    CD_wave = (1.0 - w) * CD_dd + w * CD_ackeret

    CD_skin = 0.006                       # skin friction estimate
    CD      = CD_skin + CD_wave

    # Mach (cone) angle — undefined for M <= 1
    mu = np.degrees(np.arcsin(np.clip(1.0 / M, 0.0, 1.0))) if M > 1.0 else None

    # Reynolds number: Re = rho * U * L / mu_dyn  (L=1 m, ISA sea-level)
    a_sl = 340.29  # m/s, speed of sound at sea level ISA
    U_ms = M * a_sl
    mu_dyn = 1.789e-5   # Pa·s at sea level
    Re   = rho0 * U_ms * 1.0 / mu_dyn * 1e-6   # millions

    q    = 0.5 * rho0 * U_ms**2

    return dict(CD=CD, CD_wave=CD_wave, Cpmin=Cpmin,
                shock_angle=mu, Re=Re, q=q, beta=beta,
                stag_pressure_ratio=stag_ratio,
                M_dd=M_dd, tc=tc)