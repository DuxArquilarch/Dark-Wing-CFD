# ================================================================= #
# ns_window.py — Tk popup running the 2D Navier-Stokes solver       #
# ================================================================= #

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from theme import BG, PANEL, ACCENT, TEXT
from ns_solver import ns_step, profile_to_mask

class NavierStokesWindow:
    """Popup window running the 2D N-S solver on an airfoil cross-section."""
    def __init__(self, parent, profile_pts, profile_name):
        self.win = tk.Toplevel(parent)
        self.win.title(f"N-S 2D — {profile_name}")
        self.win.configure(bg=BG)
        self.win.geometry("900x640")

        self._pts  = profile_pts.copy()
        self._name = profile_name

        ctrl = tk.Frame(self.win, bg=PANEL, padx=8, pady=6)
        ctrl.pack(fill="x")

        tk.Label(ctrl, text="AoA (°)", bg=PANEL, fg=TEXT,
                 font=("Courier", 8)).pack(side="left")
        self.aoa_var = tk.DoubleVar(value=4.0)
        ttk.Scale(ctrl, from_=-15, to=25, variable=self.aoa_var,
                  orient="horizontal", length=140).pack(side="left", padx=6)

        tk.Label(ctrl, text="Re", bg=PANEL, fg=TEXT,
                 font=("Courier", 8)).pack(side="left", padx=(12, 0))
        self.re_var = tk.DoubleVar(value=500.0)
        ttk.Scale(ctrl, from_=50, to=2000, variable=self.re_var,
                  orient="horizontal", length=140).pack(side="left", padx=6)

        tk.Button(ctrl, text="RUN", command=self._run,
                  bg=ACCENT, fg=BG, font=("Courier", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="left", padx=10)

        self.fig = Figure(figsize=(9, 5.5), dpi=90)
        self.fig.patch.set_facecolor(BG)
        cv = FigureCanvasTkAgg(self.fig, master=self.win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        self.canvas = cv

        self._run()

    def _run(self):
        Nx, Ny = 200, 120
        aoa = float(self.aoa_var.get())
        Re  = max(10.0, float(self.re_var.get()))

        mask  = profile_to_mask(self._pts, aoa, Nx, Ny)
        v_in  = 1.0
        rho   = 1.0
        mu    = rho * v_in * 1.0 / Re     # chord = 1 unit
        dl    = 1.0 / Nx

        vx = np.ones((Nx, Ny)) * v_in
        vy = np.zeros((Nx, Ny))
        p  = np.zeros((Nx, Ny))
        vx[mask == 1] = 0.0
        vy[mask == 1] = 0.0

        for _ in range(300):
            vx, vy, p, _, _ = ns_step(vx, vy, p, mask, rho, mu, dl, 1e-3, v_in)

        vmag = np.sqrt(vx**2 + vy**2)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.set_title(f"{self._name}  AoA={aoa:.1f}°  Re={Re:.0f}  — velocity magnitude",
                     color=ACCENT, fontsize=8, fontfamily="monospace")
        im = ax.imshow(vmag.T, origin="lower", cmap="turbo",
                       aspect="auto", interpolation="bicubic")
        ax.imshow(np.where(mask.T, 1.0, np.nan), origin="lower",
                  cmap="gray", aspect="auto", alpha=0.8)
        self.fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label(
            "|V|", color=ACCENT, fontsize=7)
        self.canvas.draw()


