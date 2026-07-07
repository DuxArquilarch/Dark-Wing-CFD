# ================================================================= #
# gui.py — Tk/Matplotlib main application window (CFDVisualizer)    #
# ================================================================= #

import os
import math
import warnings

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
import scipy.ndimage as ndimage

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from theme import (BG, PANEL, ACCENT, TEXT, MUTED, GREEN, YELLOW, RED,
                    BORDER, CRYSTAL_HIGH, CRYSTAL_MID, CRYSTAL_DIM, CMAPS)
from geometry import parse_stl, parse_step, load_geometry, _NACA2412
from wing_presets import WING_SHAPES, _imported_stl_planform, _close
from flow_field import FlowField, aero_stats
from ns_window import NavierStokesWindow

warnings.filterwarnings("ignore")

class CFDVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("◈ Dark Wing CFD ◈")
        root.configure(bg=BG)
        root.minsize(1300, 840)

        self.field       = FlowField(180, 140)
        self._exporting_gif = False
        self._solve_job  = None
        self._wing_name  = "Tapered Swept"
        self._rotation_deg = 0          # current planform rotation (0/90/180/270)
        self._poly_x, self._poly_y = WING_SHAPES[self._wing_name]()
        self._profile_pts  = _NACA2412.copy()
        self._profile_name = "NACA2412 (default)"

        self._ns_wins = []
        self._build_ui()
        self._solve_and_render()

    # ── UI builder ────────────────────────────────────────────────
    def _build_ui(self):
        lf_outer = tk.Frame(self.root, bg=BORDER, width=290)
        lf_outer.pack(side="left", fill="y", padx=(6, 0), pady=6)
        lf_outer.pack_propagate(False)

        lf = tk.Frame(lf_outer, bg=CRYSTAL_DIM, width=288)
        lf.pack(fill="both", expand=True, padx=1, pady=1)
        lf.pack_propagate(False)

        sb_canvas = tk.Canvas(lf, bg=CRYSTAL_DIM, highlightthickness=0)
        sb_scroll = tk.Scrollbar(lf, orient="vertical", command=sb_canvas.yview)
        sb_scroll.pack(side="right", fill="y")
        sb_canvas.pack(side="left", fill="both", expand=True)
        sb_canvas.configure(yscrollcommand=sb_scroll.set)

        inner = tk.Frame(sb_canvas, bg=CRYSTAL_DIM)
        inner_id = sb_canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",    lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))
        sb_canvas.bind("<Configure>", lambda e: sb_canvas.itemconfig(inner_id, width=e.width))
        sb_canvas.bind_all("<MouseWheel>", lambda e: sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Header
        hdr = tk.Frame(inner, bg=PANEL, pady=6)
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="◈  DARK WING  CFD", bg=PANEL, fg=ACCENT,
                 font=("Courier", 10, "bold")).pack()
        tk.Label(hdr, text="Euler + Navier-Stokes Visualizer", bg=PANEL, fg=TEXT,
                 font=("Courier", 7)).pack()

        def sep():
            tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=2)

        def sec(t, icon="▸"):
            sep()
            sf = tk.Frame(inner, bg=CRYSTAL_MID, pady=3); sf.pack(fill="x")
            tk.Label(sf, text=f" {icon} {t}", bg=CRYSTAL_MID, fg=ACCENT,
                     font=("Courier", 8, "bold"), anchor="w").pack(fill="x", padx=6)

        def crystal_btn(parent, text, cmd, color=ACCENT):
            b = tk.Button(parent, text=text, bg=PANEL, fg=color,
                          font=("Courier", 9, "bold"), relief="flat",
                          bd=0, padx=8, pady=4, cursor="hand2",
                          activebackground=CRYSTAL_MID,
                          activeforeground=CRYSTAL_HIGH, command=cmd)
            b.pack(fill="x", padx=6, pady=(4, 2))
            return b

        def crystal_check(parent, text, variable, cmd):
            f = tk.Frame(parent, bg=CRYSTAL_DIM, cursor="hand2"); f.pack(fill="x", padx=6, pady=1)
            ind = tk.Label(f, text="□", bg=CRYSTAL_DIM, fg=BORDER, font=("Courier", 8), width=2)
            ind.pack(side="left")
            lbl = tk.Label(f, text=text, bg=CRYSTAL_DIM, fg=MUTED, font=("Courier", 8), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            def _toggle():
                variable.set(not variable.get()); cmd()
                ind.config(text="◈" if variable.get() else "□",
                           fg=ACCENT if variable.get() else BORDER)
                lbl.config(fg=CRYSTAL_HIGH if variable.get() else MUTED)
            for w in [f, ind, lbl]:
                w.bind("<Button-1>", lambda e: _toggle())
            ind.config(text="◈" if variable.get() else "□",
                       fg=ACCENT if variable.get() else BORDER)
            lbl.config(fg=CRYSTAL_HIGH if variable.get() else MUTED)
            return ind, lbl

        def slider(label, key, mn, mx, init, fmt="{:.2f}"):
            var = tk.DoubleVar(value=init); setattr(self, key, var)
            row = tk.Frame(inner, bg=CRYSTAL_DIM, pady=1); row.pack(fill="x", padx=6)
            top = tk.Frame(row, bg=CRYSTAL_DIM); top.pack(fill="x")
            tk.Label(top, text=label, bg=CRYSTAL_DIM, fg=MUTED,
                     font=("Courier", 8), anchor="w").pack(side="left")
            lv = tk.Label(top, text=fmt.format(init), bg=CRYSTAL_DIM,
                          fg=ACCENT, font=("Courier", 8, "bold"), width=7)
            lv.pack(side="right")
            def _upd(val, _lv=lv, _fmt=fmt):
                _lv.config(text=_fmt.format(float(val)))
                if self._solve_job is not None: self.root.after_cancel(self._solve_job)
                self._solve_job = self.root.after(140, self._solve_and_render)
            ttk.Scale(row, from_=mn, to=mx, variable=var, orient="horizontal",
                      command=_upd, style="Crystal.Horizontal.TScale").pack(fill="x", pady=(0, 2))

        # ── Geometry (STL / STEP / DAT) ──────────────────────────
        sec("Geometry (.DAT / .STL / .STEP)", "◈")
        self.af_path_var = tk.StringVar(value="(none — using NACA 2412)")
        path_row = tk.Frame(inner, bg=CRYSTAL_DIM); path_row.pack(fill="x", padx=6, pady=4)
        path_entry = tk.Entry(path_row, textvariable=self.af_path_var,
                              bg=PANEL, fg=TEXT, font=("Courier", 7),
                              relief="flat", state="readonly")
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        def _browse_geo():
            path = filedialog.askopenfilename(
                title="Select Geometry",
                filetypes=[
                    ("All supported", "*.dat *.DAT *.stl *.STL *.stlb *.STLB *.step *.STEP *.stp *.STP"),
                    ("Airfoil DAT",   "*.dat *.DAT"),
                    ("STL 3D",        "*.stl *.STL *.stlb"),
                    ("STEP/STP CAD",  "*.step *.STEP *.stp *.STP"),
                    ("All files",     "*.*"),
                ])
            if not path:
                return
            try:
                pts, name = load_geometry(path)
                self._profile_pts  = pts
                self._profile_name = name
                self.af_path_var.set(os.path.basename(path))
                self.af_name_lbl.config(text=name, fg=GREEN)
                # Also update planform if it's a 3D file (build planform from bounding box)
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.stl', '.stlb', '.step', '.stp'):
                    self._update_planform_from_3d(path, ext)
            except Exception as ex:
                messagebox.showerror("Geometry Load Error", str(ex))

        tk.Button(path_row, text="OPEN", command=_browse_geo,
                  bg=PANEL, fg=ACCENT, font=("Courier", 8, "bold"),
                  relief="flat", padx=6, cursor="hand2").pack(side="right")

        self.af_name_lbl = tk.Label(inner, text=self._profile_name,
                                    bg=CRYSTAL_DIM, fg=GREEN,
                                    font=("Courier", 7), padx=8, anchor="w")
        self.af_name_lbl.pack(fill="x")

        # ── Wing Shape (planform) ─────────────────────────────────
        sec("Planform Wing Shape", "◈")
        self.wing_var = tk.StringVar(value=self._wing_name)
        wing_indicators = {}

        def _refresh_wing():
            for nm, (ind, lbl) in wing_indicators.items():
                sel = self.wing_var.get() == nm
                ind.config(text="◆" if sel else "◇",
                           fg=ACCENT if sel else CRYSTAL_MID)
                lbl.config(fg=CRYSTAL_HIGH if sel else MUTED)

        for name in WING_SHAPES:
            f = tk.Frame(inner, bg=CRYSTAL_DIM, cursor="hand2"); f.pack(fill="x", padx=6, pady=1)
            ind = tk.Label(f, text="◇", bg=CRYSTAL_DIM, fg=CRYSTAL_MID, font=("Courier", 7), width=2)
            ind.pack(side="left")
            lbl = tk.Label(f, text=name, bg=CRYSTAL_DIM, fg=MUTED, font=("Courier", 8), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            wing_indicators[name] = (ind, lbl)
            def _sel(n=name):
                self.wing_var.set(n); self._load_wing(); _refresh_wing()
            for w in [f, ind, lbl]:
                w.bind("<Button-1>", lambda e, fn=_sel: fn())

        self.planform_lbl = tk.Label(inner, text=self._wing_name, bg=CRYSTAL_DIM,
                                     fg=GREEN, font=("Courier", 7), padx=8)
        self.planform_lbl.pack(anchor="w")
        _refresh_wing()

        # ── Planform Rotation ─────────────────────────────────────
        sec("Planform Rotation", "↻")
        rot_row = tk.Frame(inner, bg=CRYSTAL_DIM); rot_row.pack(fill="x", padx=6, pady=4)
        self._rot_btns = {}
        for angle, label in [(0, "  0°  "), (90, " 90°  "), (180, "180°  "), (270, "270°  ")]:
            is_active = (angle == self._rotation_deg)
            btn = tk.Button(
                rot_row, text=label,
                bg=CRYSTAL_MID if is_active else PANEL,
                fg=CRYSTAL_HIGH if is_active else MUTED,
                font=("Courier", 8, "bold"), relief="flat",
                bd=0, padx=4, pady=3, cursor="hand2",
                activebackground=CRYSTAL_MID, activeforeground=CRYSTAL_HIGH,
                command=lambda a=angle: self._rotate_planform(a),
            )
            btn.pack(side="left", fill="x", expand=True, padx=1)
            self._rot_btns[angle] = btn
        tk.Label(inner, text="rotate wing/STL planform by fixed 90° steps",
                 bg=CRYSTAL_DIM, fg=BORDER, font=("Courier", 6), padx=8).pack(anchor="w")

        # ── Mach GIF ─────────────────────────────────────────────
        sec("Mach Sweep GIF", "⊳")
        gif_grid = tk.Frame(inner, bg=CRYSTAL_DIM); gif_grid.pack(fill="x", padx=6, pady=4)
        self.gif_start = tk.StringVar(value="0.3")
        self.gif_stop  = tk.StringVar(value="1.8")
        self.gif_step  = tk.StringVar(value="0.05")
        self.gif_fps   = tk.StringVar(value="5")
        self.gif_hold  = tk.StringVar(value="1")
        for i, (lt, var, w) in enumerate([
                ("from", self.gif_start, 4), ("to", self.gif_stop, 4),
                ("step", self.gif_step, 4),  ("fps", self.gif_fps, 3),
                ("hold", self.gif_hold, 3)]):
            tk.Label(gif_grid, text=lt, bg=CRYSTAL_DIM, fg=MUTED,
                     font=("Courier", 7)).grid(row=0, column=i, sticky="w", padx=2)
            tk.Entry(gif_grid, textvariable=var, bg=PANEL, fg=ACCENT,
                     font=("Courier", 8), width=w, relief="flat",
                     highlightthickness=1, highlightbackground=ACCENT,
                     insertbackground=ACCENT).grid(row=1, column=i, padx=(0,3), pady=2, sticky="ew")
        self.gif_btn = crystal_btn(inner, "⊳  Export Mach GIF", self._export_gif, ACCENT)
        self.gif_status = tk.Label(inner, text="", bg=CRYSTAL_DIM, fg=GREEN,
                                   font=("Courier", 7), anchor="w", padx=8)
        self.gif_status.pack(fill="x")

        # ── Flow Parameters ───────────────────────────────────────
        sec("Flow Parameters (Euler/Panel)", "≋")
        slider("Mach", "v_mach",  0.10, 5.00, 0.70)
        slider("γ",    "v_gamma", 1.10, 1.67, 1.40)
        slider("ρ₀",   "v_rho",   0.20, 3.00, 1.225)

        # ── Scalar Field ──────────────────────────────────────────
        sec("Scalar Field", "⬡")
        self.mode_var = tk.StringVar(value="pressure")
        self.cmap_var = tk.StringVar(value="turbo")

        cmap_row = tk.Frame(inner, bg=CRYSTAL_DIM, pady=2); cmap_row.pack(fill="x", padx=6)
        tk.Label(cmap_row, text="Colormap", bg=CRYSTAL_DIM, fg=MUTED,
                 font=("Courier", 7), anchor="w").pack(anchor="w")
        cm = tk.OptionMenu(cmap_row, self.cmap_var, *CMAPS.keys(), command=self._render)
        cm.config(bg=PANEL, fg=ACCENT, font=("Courier", 8), relief="flat",
                  highlightthickness=1, highlightbackground=ACCENT, width=12)
        cm["menu"].config(bg=PANEL, fg=TEXT, activebackground=CRYSTAL_MID,
                          activeforeground=CRYSTAL_HIGH)
        cm.pack(fill="x", pady=(2, 6))

        mode_indicators = {}
        def _refresh_mode():
            for nm, (ind, lbl) in mode_indicators.items():
                sel = self.mode_var.get() == nm
                ind.config(text="◆" if sel else "◇", fg=ACCENT if sel else CRYSTAL_MID)
                lbl.config(fg=CRYSTAL_HIGH if sel else MUTED)

        for m in ["pressure","mach","vorticity","temperature"]:
            f = tk.Frame(inner, bg=CRYSTAL_DIM, cursor="hand2"); f.pack(fill="x", padx=6, pady=1)
            ind = tk.Label(f, text="◇", bg=CRYSTAL_DIM, fg=CRYSTAL_MID, font=("Courier", 7), width=2)
            ind.pack(side="left")
            lbl = tk.Label(f, text=m, bg=CRYSTAL_DIM, fg=MUTED, font=("Courier", 8), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            mode_indicators[m] = (ind, lbl)
            def _sel_m(nm=m):
                self.mode_var.set(nm); self._render(); _refresh_mode()
            for w in [f, ind, lbl]:
                w.bind("<Button-1>", lambda e, fn=_sel_m: fn())

        _refresh_mode()

        # ── Render Layers ─────────────────────────────────────────
        sec("Render Layers", "◎")
        self.lay_stream = tk.BooleanVar(value=True)
        self.lay_shock  = tk.BooleanVar(value=True)
        self.lay_body   = tk.BooleanVar(value=True)
        self.lay_glow   = tk.BooleanVar(value=True)
        self.lay_vec    = tk.BooleanVar(value=False)
        self.lay_iso    = tk.BooleanVar(value=False)
        for lt, var in [("Flow traces", self.lay_stream), ("Shock wave", self.lay_shock),
                         ("Wing body",   self.lay_body),  ("Crystal glow", self.lay_glow),
                         ("Mesh grid",   self.lay_vec),   ("Isobars",     self.lay_iso)]:
            crystal_check(inner, lt, var, self._render)

        # ── Flow Analytics ────────────────────────────────────────
        sec("Flow Analytics", "⊞")
        stats_panel = tk.Frame(inner, bg=PANEL, pady=6, padx=8); stats_panel.pack(fill="x", padx=6, pady=4)
        self.stat_labels = {}
        for k in ["CD", "Cp_min", "Shock°", "Re×10⁶", "q", "β"]:
            row = tk.Frame(stats_panel, bg=PANEL); row.pack(fill="x", pady=1)
            tk.Label(row, text=k, bg=PANEL, fg=TEXT,
                     font=("Courier", 8), width=10, anchor="w").pack(side="left")
            lv = tk.Label(row, text="—", bg=PANEL, fg=GREEN, font=("Courier", 8, "bold"))
            lv.pack(side="right")
            self.stat_labels[k] = lv

        sep()
        crystal_btn(inner, "↺  RECOMPUTE", self._solve_and_render, GREEN)
        tk.Frame(inner, bg=CRYSTAL_DIM, height=10).pack()

        # ── Main canvas area ──────────────────────────────────────
        rf = tk.Frame(self.root, bg=BG)
        rf.pack(side="right", fill="both", expand=True, padx=(4, 6), pady=6)

        title_bar = tk.Frame(rf, bg=PANEL, pady=4); title_bar.pack(fill="x", pady=(0, 4))
        tk.Label(title_bar, text="◈  WING PLANFORM  EULER + N-S FLOW",
                 bg=PANEL, fg=ACCENT, font=("Courier", 9, "bold")).pack(side="left", padx=8)
        tk.Label(title_bar, text="[TOP VIEW]", bg=PANEL, fg=TEXT,
                 font=("Courier", 7)).pack(side="right", padx=8)

        self.fig = Figure(figsize=(10, 7), dpi=96)
        self.fig.patch.set_facecolor(BG)

        cv_frame = tk.Frame(rf, bg=BORDER, padx=1, pady=1); cv_frame.pack(fill="both", expand=True)
        inner_cv = tk.Frame(cv_frame, bg=BG); inner_cv.pack(fill="both", expand=True)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=inner_cv)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas_widget, rf)
        tb.config(bg=PANEL, relief="flat"); tb.update()

    # ── Geometry helpers ──────────────────────────────────────────────
    def _update_planform_from_3d(self, path, ext):
        """
        Build top-view planform polygon directly from the 3-D external silhouette.
        """
        try:
            if ext in ('.stl', '.stlb'):
                tri = parse_stl(path)
                verts = np.vstack(tri).astype(np.float64)
            else:
                tri = None
                verts = parse_step(path).astype(np.float64)

            ranges = verts.max(axis=0) - verts.min(axis=0)
            order = np.argsort(ranges)[::-1]
            # order[0] = longest 3-D axis → wing SPAN (maps to Y in planform)
            # order[1] = second  axis     → wing CHORD (maps to X in planform)
            span_ax, chord_ax = int(order[0]), int(order[1])

            if tri is not None:
                tri2 = np.stack([
                    tri[0][:, [chord_ax, span_ax]],
                    tri[1][:, [chord_ax, span_ax]],
                    tri[2][:, [chord_ax, span_ax]],
                ], axis=1)
                pts2_all = tri2.reshape(-1, 2)
            else:
                tri2 = None
                pts2_all = verts[:, [chord_ax, span_ax]]

            mn = pts2_all.min(axis=0)
            mx = pts2_all.max(axis=0)
            span_raw = float(mx[1] - mn[1])
            chord_raw = float(mx[0] - mn[0])
            if span_raw < 1e-9 or chord_raw < 1e-9:
                raise ValueError("Projected geometry is too thin.")

            if HAS_CV2:
                target = 1000.0
                pad = 24
                pix_scale = target / max(span_raw, chord_raw)
                w = int(np.ceil(chord_raw * pix_scale)) + pad * 2 + 3
                h = int(np.ceil(span_raw * pix_scale)) + pad * 2 + 3
                mask = np.zeros((h, w), dtype=np.uint8)

                def _to_pix(a):
                    out = np.empty_like(a, dtype=np.float64)
                    out[..., 0] = (a[..., 0] - mn[0]) * pix_scale + pad
                    out[..., 1] = (mx[1] - a[..., 1]) * pix_scale + pad
                    return np.rint(out).astype(np.int32)

                if tri2 is not None:
                    for poly in _to_pix(tri2):
                        cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
                else:
                    pix = _to_pix(pts2_all)
                    cv2.fillPoly(mask, [cv2.convexHull(pix)], 255, lineType=cv2.LINE_AA)

                k = max(3, int(round(target / 320.0)))
                kernel = np.ones((k, k), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    raise ValueError("No external STL silhouette found.")

                contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
                if len(contour) > 650:
                    step = int(math.ceil(len(contour) / 650))
                    contour = contour[::step]

                raw_x = (contour[:, 0] - pad) / pix_scale + mn[0]
                raw_y = mx[1] - (contour[:, 1] - pad) / pix_scale
            else:
                hull_pts = pts2_all
                c = hull_pts.mean(axis=0)
                ang = np.arctan2(hull_pts[:, 1] - c[1], hull_pts[:, 0] - c[0])
                keep = np.argsort(ang)
                raw_x = hull_pts[keep, 0]
                raw_y = hull_pts[keep, 1]

            # ── Orient so that CHORD is along X and SPAN along Y ─────
            # After projection raw_x = chord axis candidate, raw_y = span.
            # But if the STL was modelled with a different axis convention,
            # the span can end up larger in raw_x.  Detect and swap.
            raw_x_range = float(raw_x.max() - raw_x.min())
            raw_y_range = float(raw_y.max() - raw_y.min())
            if raw_x_range > raw_y_range:
                # raw_x is the longer axis → treat as span, swap so span→Y
                raw_x, raw_y = raw_y, raw_x

            span_mid  = 0.5 * (float(raw_y.max()) + float(raw_y.min()))
            chord_mid = 0.5 * (float(raw_x.max()) + float(raw_x.min()))
            span  = max(float(raw_y.max() - raw_y.min()), 1e-9)
            scale = 2.4 / span
            px = (raw_x - chord_mid) * scale
            py = (raw_y - span_mid)  * scale

            # Ensure nose (leading edge) is at negative X so it faces the inlet
            # (flow enters from left: vx[0,:] = U_inf).
            centre_band = np.abs(py) <= max(0.08, 0.04 * max(1.0, float(np.max(np.abs(py)))))
            if np.any(centre_band):
                nose_x = float(px[centre_band].min())
                tail_x = float(px[centre_band].max())
                # Flip so the SMALLER (more negative) end faces the inlet
                if abs(tail_x) > abs(nose_x):
                    px = -px

            px, py = _close(px.astype(np.float64), py.astype(np.float64))

            _imported_stl_planform[0] = (px, py)
            self._poly_x, self._poly_y = px, py
            self._wing_name = self._profile_name + " (from file)"
            self.planform_lbl.config(text=self._wing_name[:30], fg=YELLOW)
            self._rotation_deg = 0
            if hasattr(self, '_rot_btns'):
                for angle, btn in self._rot_btns.items():
                    btn.config(fg=CRYSTAL_HIGH if angle == 0 else MUTED,
                               bg=CRYSTAL_MID  if angle == 0 else PANEL)
            self._solve_and_render()

        except Exception as ex:
            messagebox.showwarning(
                "Planform from STL",
                f"Could not build planform from 3-D file:\n{ex}\n\n"
                "Using current wing shape instead.")

    def _open_ns_window(self):
        win = NavierStokesWindow(self.root, self._profile_pts, self._profile_name)
        self._ns_wins.append(win)

    # ── Planform solver ───────────────────────────────────────────────
    def _load_wing(self, *_):
        name = self.wing_var.get()
        if name == "Imported STL" and _imported_stl_planform[0] is None:
            messagebox.showinfo(
                "No STL loaded",
                "Open a .STL or .STEP file first via the Geometry section.\n"
                "The planform will be built automatically from that file.")
            # Revert selector to current wing
            self.wing_var.set(self._wing_name)
            return
        self._poly_x, self._poly_y = WING_SHAPES[name]()
        self._wing_name = name
        self.planform_lbl.config(text=name, fg=GREEN)
        if name == "Ballistic" and hasattr(self, 'cmap_var'):
            self.cmap_var.set("schlieren_gray")
        # Reset rotation to 0 for freshly-loaded shape
        self._rotation_deg = 0
        if hasattr(self, '_rot_btns'):
            for angle, btn in self._rot_btns.items():
                btn.config(fg=CRYSTAL_HIGH if angle == 0 else MUTED,
                           bg=CRYSTAL_MID  if angle == 0 else PANEL)
        self._solve_and_render()

    def _rotate_planform(self, deg):
        """Rotate planform polygon by deg degrees (0/90/180/270) and re-solve."""
        self._rotation_deg = deg % 360
        # Get base (unrotated) planform from current wing shape
        name = self.wing_var.get()
        if name == "Imported STL" and _imported_stl_planform[0] is not None:
            bx, by = _imported_stl_planform[0]
        elif name in WING_SHAPES:
            bx, by = WING_SHAPES[name]()
        else:
            bx, by = self._poly_x.copy(), self._poly_y.copy()

        if deg == 0:
            self._poly_x, self._poly_y = bx, by
        else:
            rad = math.radians(deg)
            c, s = math.cos(rad), math.sin(rad)
            cx = (bx.max() + bx.min()) / 2.0
            cy = (by.max() + by.min()) / 2.0
            xc = bx - cx
            yc = by - cy
            rx = c * xc - s * yc + cx
            ry = s * xc + c * yc + cy
            self._poly_x, self._poly_y = rx, ry

        # Update button states
        for angle, btn in self._rot_btns.items():
            btn.config(
                fg=CRYSTAL_HIGH if angle == self._rotation_deg else MUTED,
                bg=CRYSTAL_MID  if angle == self._rotation_deg else PANEL,
            )
        self._solve_and_render()

    def _solve_and_render(self, *_):
        self._solve_job = None
        mach  = float(self.v_mach.get())
        gamma = float(self.v_gamma.get())
        rho0  = float(self.v_rho.get())
        self.field.solve(mach, gamma, rho0, self._poly_x, self._poly_y)
        st = aero_stats(mach, gamma, rho0)
        self.stat_labels["CD"].config(text=f"{st['CD']:.4f}")
        self.stat_labels["Cp_min"].config(text=f"{st['Cpmin']:.3f}")
        self.stat_labels["Shock°"].config(
            text=f"{st['shock_angle']:.1f}" if st['shock_angle'] else "—")
        self.stat_labels["Re×10⁶"].config(text=f"{st['Re']:.2f}")
        self.stat_labels["q"].config(text=f"{st['q']:.3f}")
        self.stat_labels["β"].config(text=f"{st['beta']:.3f}")
        self._render()

    # ── Render helpers ────────────────────────────────────────────────
    def _field_norm(self, S, mode):
        fin = S[np.isfinite(S)]
        if fin.size == 0: return Normalize(0, 1), 0.0
        lo, hi = np.nanpercentile(fin, [2, 98])
        if abs(hi - lo) < 1e-12: lo -= 0.5; hi += 0.5
        if mode == "vorticity" and lo < 0 < hi:
            lim = max(abs(lo), abs(hi))
            return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), lo
        return Normalize(vmin=lo, vmax=hi), lo

    def _sample_vel(self, x, y):
        f = self.field
        if not (f.xi[0] < x < f.xi[-1] and f.yi[0] < y < f.yi[-1]):
            return None
        ix = int(np.searchsorted(f.xi, x) - 1)
        iy = int(np.searchsorted(f.yi, y) - 1)
        ix = np.clip(ix, 0, f.NX - 2)
        iy = np.clip(iy, 0, f.NY - 2)
        if f._mask[ix, iy]: return None
        tx = (x - f.xi[ix]) / (f.xi[ix+1] - f.xi[ix] + 1e-14)
        ty = (y - f.yi[iy]) / (f.yi[iy+1] - f.yi[iy] + 1e-14)
        def bi(A):
            return ((1-tx)*(1-ty)*A[ix,iy] + tx*(1-ty)*A[ix+1,iy] +
                    (1-tx)*ty*A[ix,iy+1]   + tx*ty*A[ix+1,iy+1])
        return bi(f.U), bi(f.V)

    def _draw_streaks(self, ax, cmap, alpha=0.65):
        f = self.field
        lines, speeds = [], []

        # Leading-edge X position — seed upstream of it for frontal flow entry
        nose_x    = float(np.min(self._poly_x))
        half_span = float(np.max(np.abs(self._poly_y)))

        # Primary seed column: left inlet wall (always frontal, chord-wise)
        seed_x_inlet = f.xi[2]
        seed_ys = np.linspace(f.yi[2], f.yi[-3], 60)
        seeds = [(seed_x_inlet, y) for y in seed_ys]

        # Secondary column: just upstream of the leading edge for tight nose traces
        upstream_x = max(seed_x_inlet, nose_x - 0.20)
        for y0 in np.linspace(-half_span * 1.25, half_span * 1.25, 45):
            seeds.append((upstream_x, y0))

        # Integration parameters — use actual velocity (not normalised) for
        # physically correct curvature around the body
        max_steps = 220
        ds = 0.018          # arc-length step in world units

        for (x0, y0) in seeds:
            ix0 = int(np.argmin(np.abs(f.xi - x0)))
            iy0 = int(np.argmin(np.abs(f.yi - y0)))
            ix0 = np.clip(ix0, 0, f.NX - 1)
            iy0 = np.clip(iy0, 0, f.NY - 1)
            if f._mask[ix0, iy0]:
                continue

            pts = [(x0, y0)]; x, y = x0, y0; local_spd = []
            for _ in range(max_steps):
                vel = self._sample_vel(x, y)
                if vel is None:
                    break
                u, v = vel
                spd = float(np.hypot(u, v))
                if spd < 1e-9:
                    break
                local_spd.append(spd)
                # Advance by fixed arc-length in actual flow direction
                x += (u / spd) * ds
                y += (v / spd) * ds
                pts.append((x, y))

            if len(pts) >= 3:
                lines.append(pts)
                speeds.append(float(np.mean(local_spd)) if local_spd else 0.0)

        if not lines:
            return

        lo, hi = np.percentile(speeds, [5, 95])
        if abs(hi - lo) < 1e-12:
            lo -= 0.5; hi += 0.5
        lc = LineCollection(lines, cmap=cmap, norm=Normalize(vmin=lo, vmax=hi),
                            linewidths=0.55, alpha=alpha, capstyle="round", zorder=3)
        lc.set_array(np.asarray(speeds))
        ax.add_collection(lc)

    def _draw_mesh(self, ax, cmap):
        f = self.field; sk = 9
        ml  = [[(x, f.yi[0]), (x, f.yi[-1])] for x in f.xi[::sk]]
        ml += [[(f.xi[0], y), (f.xi[-1], y)] for y in f.yi[::sk]]
        ax.add_collection(LineCollection(ml, colors=ACCENT, linewidths=0.18,
                                         alpha=0.25, zorder=2))
        self._draw_streaks(ax, cmap, alpha=0.80)

    def _render(self, *_, mach_override=None, draw=True):
        f    = self.field
        mode = self.mode_var.get()
        S    = f.scalar(mode)
        mach = mach_override if mach_override is not None else float(self.v_mach.get())
        cmap = CMAPS.get(self.cmap_var.get(), "pp")

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        S_disp   = S.T
        S_masked = np.where(f._mask.T, np.nan, S_disp)
        norm, fill_val = self._field_norm(S_masked, mode)
        extent = [f.xi[0], f.xi[-1], f.yi[0], f.yi[-1]]

        ax.imshow(S_masked, origin='lower', extent=extent,
                  cmap=cmap, norm=norm, aspect='auto',
                  interpolation='bicubic', zorder=0)

        if self.lay_glow.get():
            Sg = ndimage.gaussian_filter(np.nan_to_num(S_disp, nan=fill_val), sigma=5)
            ax.imshow(Sg, origin='lower', extent=extent, cmap=cmap,
                      norm=Normalize(Sg.min(), Sg.max()),
                      aspect='auto', interpolation='bicubic', alpha=0.35, zorder=1)

        if self.lay_stream.get():
            try:
                self._draw_streaks(ax, cmap)
            except Exception:
                self.lay_stream.set(False)

        if self.lay_vec.get():
            try:
                self._draw_mesh(ax, cmap)
            except Exception:
                self.lay_vec.set(False)

        if self.lay_iso.get():
            lv = np.linspace(np.nanpercentile(f.P, 5), np.nanpercentile(f.P, 95), 14)
            ax.contour(f.xi, f.yi, f.P.T, levels=lv,
                       colors=YELLOW, linewidths=0.5, alpha=0.55, zorder=4)



        if self.lay_body.get():
            px = np.append(self._poly_x, self._poly_x[0])
            py = np.append(self._poly_y, self._poly_y[0])
            ax.fill(px, py, color="#02060f", zorder=6)
            ax.plot(px, py, color=ACCENT, linewidth=2.0, zorder=7, alpha=0.9)
            ax.plot(px, py, color=CRYSTAL_HIGH, linewidth=0.6, zorder=8, alpha=0.5)
            x0 = f.xi[0] + 0.15
            ax.annotate("", xy=(x0 + 0.4, 0.0), xytext=(x0, 0.0), zorder=9,
                        arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                        lw=1.3, mutation_scale=12))
            ax.text(x0 + 0.44, 0.07, "U∞", color=GREEN,
                    fontsize=7, fontfamily="monospace", zorder=9)

        if self.lay_shock.get() and mach > 1.0:
            # Mach cone (Mach lines): mu = arcsin(1/M), measured from the
            # free-stream direction (+x). Only exists for M > 1.
            mu = math.asin(min(1.0, 1.0 / mach))

            # Anchor at the body's nose (most-upstream / min-x point).
            i_nose = int(np.argmin(self._poly_x))
            x_nose = float(self._poly_x[i_nose])
            y_nose = float(self._poly_y[i_nose])

            x_end  = f.xi[-1]
            dx     = max(x_end - x_nose, 0.0)
            dy     = dx * math.tan(mu)

            # zorder kept BELOW the body fill/outline (6/7/8) so the solid
            # geometry occludes the cone where it passes through the body —
            # the lines visually wrap around the wing instead of slicing
            # through it, matching the actual shape of the object.
            line_kw = dict(color=RED, linewidth=1.6, linestyle=(0, (5, 3)),
                           alpha=0.35, zorder=5, solid_capstyle="round",
                           clip_on=True)
            ax.plot([x_nose, x_end], [y_nose, y_nose + dy], **line_kw)
            ax.plot([x_nose, x_end], [y_nose, y_nose - dy], **line_kw)

            ax.fill_between([x_nose, x_end], [y_nose, y_nose - dy],
                             [y_nose, y_nose + dy],
                             color=RED, alpha=0.06, zorder=2, clip_on=True)

            # Clamp label position to stay within the plotted data range —
            # near M≈1, mu→90° and dy blows up, which otherwise pushes this
            # text far off-plot and makes tight_layout() shrink the whole
            # figure to "fit" it.
            lx = min(x_nose + 0.6 * dx, f.xi[-1] - 0.05)
            ly = float(np.clip(y_nose + 0.6 * dy, f.yi[0] + 0.05, f.yi[-1] - 0.05))
            ax.text(lx, ly + 0.05, f"μ={math.degrees(mu):.1f}°",
                    color=RED, fontsize=7, fontfamily="monospace",
                    zorder=11, ha="left", va="bottom", clip_on=True)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = self.fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=TEXT)
        cb.outline.set_edgecolor(ACCENT)
        cb.set_label(mode, color=ACCENT, fontsize=7, fontfamily="monospace")

        self._draw_legend(ax, mach, mode)

        ax.set_xlim(f.xi[0], f.xi[-1]); ax.set_ylim(f.yi[0], f.yi[-1])
        ax.set_xlabel("x / c  (chordwise →)", color=MUTED, fontsize=9, fontfamily="monospace")
        ax.set_ylabel("y / b  (spanwise)",    color=MUTED, fontsize=9, fontfamily="monospace")
        ax.set_title(
            f"◈  {self._wing_name}  ·  {mode}  ·  M = {mach:.3f}  [TOP VIEW]",
            color=ACCENT, fontsize=9, pad=8, fontfamily="monospace")
        ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5, linestyle=":")
        for spine in ax.spines.values():
            spine.set_edgecolor(ACCENT); spine.set_linewidth(1.2)
        ax.set_aspect('equal', 'box')
        self.fig.suptitle("◈  DARK WING CFD  ·  EULER + NAVIER-STOKES",
                          color=ACCENT, fontsize=8, y=0.999, fontfamily="monospace")
        self.fig.tight_layout(rect=[0, 0, 1, 0.999])
        if draw:
            self.canvas_widget.draw()

    # ── HUD legend ────────────────────────────────────────────────────
    def _draw_legend(self, ax, mach, mode):
        """
        Bottom-left HUD box showing live physics summary for the current frame.
        All values derived from the FlowField already solved for this Mach.
        """
        f     = self.field
        gamma = float(self.v_gamma.get())
        rho0  = float(self.v_rho.get())
        stats = aero_stats(mach, gamma, rho0)

        M     = float(mach)
        gm1   = gamma - 1.0

        # ── regime tag ─────────────────────────────────────────────────
        if M < 0.3:
            regime = "INCOMPRESSIBLE"
            reg_col = "#00e5ff"
        elif M < 0.8:
            regime = "SUBSONIC"
            reg_col = "#00e5ff"
        elif M < 1.0:
            regime = "TRANSONIC (sub)"
            reg_col = YELLOW
        elif M < 1.2:
            regime = "TRANSONIC (super)"
            reg_col = YELLOW
        elif M < 5.0:
            regime = "SUPERSONIC"
            reg_col = RED
        else:
            regime = "HYPERSONIC"
            reg_col = RED

        # ── vorticity stats (grid) ──────────────────────────────────────
        vort    = f.vorticity()
        out_mask = ~f._mask
        vort_rms = float(np.sqrt(np.mean(vort[out_mask]**2)))
        vort_pk  = float(np.max(np.abs(vort[out_mask])))

        # ── pressure extremes ───────────────────────────────────────────
        P_out   = f.P[out_mask]
        p_min   = float(np.nanmin(P_out))
        p_max   = float(np.nanmax(P_out))
        Cp_min  = stats["Cpmin"]

        # ── local Mach field stats ──────────────────────────────────────
        Mloc    = f.local_mach()[out_mask]
        M_peak  = float(np.nanpercentile(Mloc, 99))
        M_mean  = float(np.nanmean(Mloc))

        # ── speed of sound (ISA sea-level) ─────────────────────────────
        a_sl    = 340.29   # m/s
        U_ms    = M * a_sl

        # ── stagnation pressure ratio ───────────────────────────────────
        p0_ratio = stats["stag_pressure_ratio"]

        # ── wave drag / total Cd ────────────────────────────────────────
        CD_wave = stats["CD_wave"]
        CD_tot  = stats["CD"]

        # ── dynamic pressure (Pa) ───────────────────────────────────────
        q_Pa    = stats["q"]

        # ── Prandtl-Glauert correction factor ───────────────────────────
        beta    = stats["beta"]
        PG_corr = 1.0 / beta if M < 0.98 else float('nan')

        # ── shock angle (Mach cone) ─────────────────────────────────────
        shock_mu = stats["shock_angle"]   # degrees, None if subsonic

        # ── Reynolds (millions) ─────────────────────────────────────────
        Re_M    = stats["Re"]

        # ── build text lines ────────────────────────────────────────────
        lines = []
        lines.append(("▸ FLOW STATE", ACCENT, True))
        lines.append((f"  REGIME    {regime}", reg_col, False))
        lines.append((f"  M∞        {M:.3f}", TEXT, False))
        lines.append((f"  U∞        {U_ms:.1f} m/s", TEXT, False))
        lines.append((f"  Re        {Re_M:.2f} M", TEXT, False))
        lines.append(("", TEXT, False))

        lines.append(("▸ PRESSURE", ACCENT, True))
        lines.append((f"  p_min     {p_min:.4f}", "#00e5ff", False))
        lines.append((f"  p_max     {p_max:.4f}", "#f5f5f5", False))
        lines.append((f"  Cp_min    {Cp_min:.4f}", "#ffe066", False))
        lines.append((f"  q         {q_Pa:.1f} Pa", TEXT, False))
        lines.append((f"  p0/p∞     {p0_ratio:.4f}", TEXT, False))
        lines.append(("", TEXT, False))

        lines.append(("▸ VORTICITY", ACCENT, True))
        lines.append((f"  ω_rms     {vort_rms:.4f}", "#a0f0ff", False))
        lines.append((f"  |ω|_peak  {vort_pk:.4f}", "#00e5ff", False))
        lines.append(("", TEXT, False))

        lines.append(("▸ LOCAL MACH", ACCENT, True))
        lines.append((f"  M_mean    {M_mean:.4f}", TEXT, False))
        lines.append((f"  M_peak    {M_peak:.4f}", "#ffe066", False))
        lines.append(("", TEXT, False))

        lines.append(("▸ AERO COEFF", ACCENT, True))
        if M >= 0.98:
            lines.append((f"  CD_wave   {CD_wave:.5f}", RED, False))
        else:
            lines.append((f"  PG corr   1/β={PG_corr:.3f}", "#a0f0ff", False))
        lines.append((f"  CD_total  {CD_tot:.5f}", TEXT, False))
        lines.append(("", TEXT, False))

        lines.append(("▸ SHOCKWAVE", ACCENT, True))
        if shock_mu is not None:
            lines.append((f"  μ_cone    {shock_mu:.2f}°", RED, False))
            lines.append((f"  MACH CONE PRESENT", RED, False))
        else:
            lines.append((f"  No shock (M<1)", MUTED, False))

        # ── draw box in figure-fraction coords (left margin, outside axes) ──
        fig = self.fig

        # Axes bounding box in figure fraction
        ax_bbox = ax.get_position()   # Bbox(x0,y0,x1,y1) in figure fraction

        # Place legend in the left margin: from figure left edge to ax left edge
        margin_left  = 0.0
        margin_right = ax_bbox.x0
        margin_w     = margin_right - margin_left

        # box dimensions in figure fraction
        box_w_f  = margin_w * 0.82          # use 82 % of the margin width
        line_h_f = 0.0145                   # height per line in fig fraction
        box_h_f  = len(lines) * line_h_f + 0.006

        box_x0 = margin_left + (margin_w - box_w_f) * 0.5   # horizontally centred in margin
        box_y0 = ax_bbox.y0                                   # bottom-align with axes

        # background rect
        rect = FancyBboxPatch(
            (box_x0, box_y0), box_w_f, box_h_f,
            boxstyle="round,pad=0.003",
            linewidth=0.8,
            edgecolor=ACCENT,
            facecolor=BG,
            alpha=0.88,
            transform=fig.transFigure,
            zorder=19,
            clip_on=False,
        )
        fig.add_artist(rect)

        # text lines top-to-bottom
        for k, (txt, col, bold) in enumerate(lines):
            ty = box_y0 + box_h_f - (k + 1) * line_h_f + line_h_f * 0.18
            fig.text(
                box_x0 + 0.004,
                ty,
                txt,
                color=col,
                fontsize=6.0,
                fontfamily="monospace",
                fontweight="bold" if bold else "normal",
                va="bottom",
                transform=fig.transFigure,
                zorder=20,
                clip_on=False,
            )

    # ── GIF export ────────────────────────────────────────────────────
    def _gif_mach_values(self):
        start = float(self.gif_start.get().replace(",", "."))
        stop  = float(self.gif_stop.get().replace(",", "."))
        step  = abs(float(self.gif_step.get().replace(",", ".")))
        if step <= 0: raise ValueError("Step must be > 0")
        vals = np.clip(np.arange(start, stop + step*0.5, step), 0.10, 5.00)
        vals = vals[vals <= stop + 1e-9]
        if vals.size == 0: raise ValueError("Range produced no frames.")
        if vals.size > 2000: raise ValueError("Too many frames — increase step.")
        return vals

    def _export_gif(self):
        if self._exporting_gif: return
        try:
            machs = self._gif_mach_values()
            fps   = max(1, min(60, int(float(self.gif_fps.get().replace(",",".")))))
            hold  = max(1, min(120, int(float(self.gif_hold.get().replace(",",".")))))
        except Exception as ex:
            messagebox.showerror("GIF error", str(ex)); return
        path = filedialog.asksaveasfilename(
            title="Export Mach sweep GIF", defaultextension=".gif",
            initialfile=f"{self._wing_name.replace(' ','_')}_mach.gif",
            filetypes=[("GIF animation", "*.gif")])
        if not path: return
        gamma = float(self.v_gamma.get()); rho0 = float(self.v_rho.get())
        orig_mach = float(self.v_mach.get())
        total_frames = len(machs) * hold
        self._exporting_gif = True
        self.gif_btn.config(state="disabled", text="⊳  Exporting…")

        # Duration per frame in milliseconds for PIL
        frame_ms = max(1, round(1000 / fps))

        # Choose backend: PIL incremental save (no RAM accumulation) or
        # fall back to PillowWriter if PIL is not available.
        use_pil = HAS_PIL

        try:
            if use_pil:
                import io
                first_pil = None
                appended  = []

                for i, m in enumerate(machs, 1):
                    self.gif_status.config(
                        text=f"frame {i}/{len(machs)}  M={m:.2f}  "
                             f"({i*hold}/{total_frames} total)")
                    self.root.update()           # keep UI alive + progress visible

                    self.field.solve(m, gamma, rho0, self._poly_x, self._poly_y)
                    self._render(mach_override=float(m), draw=False)

                    # Render figure to PNG bytes → PIL Image (never stored as ndarray)
                    buf = io.BytesIO()
                    self.fig.savefig(buf, format="png",
                                     dpi=self.fig.dpi, facecolor=BG)
                    buf.seek(0)
                    img = _PILImage.open(buf).convert("RGB")

                    if first_pil is None:
                        first_pil = img.copy()
                    else:
                        # Append hold-copies directly to the growing GIF on disk
                        for _ in range(hold):
                            appended.append(img.copy())

                if first_pil is None:
                    raise ValueError("No frames were rendered.")

                # Write: first frame starts the file; rest appended incrementally
                first_pil.save(
                    path,
                    format="GIF",
                    save_all=True,
                    append_images=appended + [first_pil] * (hold - 1),
                    duration=frame_ms,
                    loop=0,
                    optimize=False,   # keep False — optimise pass can OOM on large GIFs
                )
            else:
                # Fallback: original PillowWriter path
                writer = PillowWriter(fps=fps)
                with writer.saving(self.fig, path, dpi=self.fig.dpi):
                    for i, m in enumerate(machs, 1):
                        self.gif_status.config(
                            text=f"frame {i}/{len(machs)}  M={m:.2f}")
                        self.root.update()
                        self.field.solve(m, gamma, rho0, self._poly_x, self._poly_y)
                        self._render(mach_override=float(m), draw=False)
                        for _ in range(hold):
                            writer.grab_frame(facecolor=BG)

            self.gif_status.config(
                text=f"✓ saved {os.path.basename(path)}  ({total_frames} frames)")
        except Exception as ex:
            messagebox.showerror("GIF error", str(ex))
            self.gif_status.config(text="export failed")
        finally:
            self.field.solve(orig_mach, gamma, rho0, self._poly_x, self._poly_y)
            self._solve_and_render()
            self.gif_btn.config(state="normal", text="⊳  Export Mach GIF")
            self._exporting_gif = False


