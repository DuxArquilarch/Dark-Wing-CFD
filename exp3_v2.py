import numpy as np
import cv2
import warnings
import tkinter as tk
from tkinter import filedialog
import os
import math
import imageio 
from numba import njit

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# FISICA E PROPRIEDADES (ISA)
# ─────────────────────────────────────────────────────────

def air_properties(altitude_m=0.0):
    """Física baseada na altitude selecionada no menu."""
    T = 288.15 - 0.0065 * altitude_m
    P = 101325.0 * (T / 288.15) ** 5.2561
    rho = P / (287.05 * T)
    mu = 1.458e-6 * T**1.5 / (T + 110.4)
    return rho, mu

def load_airfoil_robust(filepath):
    """Loader 5.0 — Suporta Airfoil Tools: Selig, Lednicer, e raw XY."""
    if not filepath or not os.path.isfile(filepath):
        # NACA 2412 fallback
        return np.array([[1,0],[0.95,0.012],[0.8,0.038],[0.6,0.060],[0.4,0.072],
                         [0.2,0.062],[0.1,0.044],[0,0],[0.1,-0.024],[0.2,-0.034],
                         [0.4,-0.040],[0.6,-0.034],[0.8,-0.018],[0.95,-0.006],[1,0]], dtype=np.float64)

    raw_lines = []
    with open(filepath, 'r', errors='ignore') as f:
        raw_lines = f.readlines()

    # --- detecta Lednicer: segunda linha tem dois floats grandes (ex: "33.  33.") ---
    def is_lednicer(lines):
        for line in lines[1:4]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                    if a > 2.0 and b > 2.0:  # contagem de pontos, não coord
                        return True
                except: pass
        return False

    coords = []

    if is_lednicer(raw_lines):
        # Formato Lednicer: pula linhas de cabeçalho (nome + contagem)
        skip = 0
        for i, line in enumerate(raw_lines):
            parts = line.split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                    if a > 2.0 and b > 2.0:
                        skip = i + 1  # pula até depois da linha de contagem
                        break
                except: pass
        for line in raw_lines[skip:]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except: continue
    else:
        # Formato Selig (padrão Airfoil Tools): pula primeira linha (nome)
        started = False
        for line in raw_lines:
            parts = line.split()
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    # pula linhas de título que acidentalmente têm 2 tokens numéricos grandes
                    if not started and abs(x) > 2.0:
                        continue
                    coords.append([x, y])
                    started = True
                except: continue

    if len(coords) < 6:
        return np.array([[1,0],[0.95,0.012],[0.8,0.038],[0.6,0.060],[0.4,0.072],
                         [0.2,0.062],[0.1,0.044],[0,0],[0.1,-0.024],[0.2,-0.034],
                         [0.4,-0.040],[0.6,-0.034],[0.8,-0.018],[0.95,-0.006],[1,0]], dtype=np.float64)

    pts = np.array(coords, dtype=np.float64)

    # NORMALIZAÇÃO 5.0 — X em [0,1], Y proporcional E centrado em 0
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    chord = x_max - x_min
    pts[:, 0] = (pts[:, 0] - x_min) / chord
    pts[:, 1] = pts[:, 1] / chord  # mantém proporção real da espessura

    return pts

# ─────────────────────────────────────────────────────────
# SOLVER ESTABILIZADO 
# ─────────────────────────────────────────────────────────

@njit
def solve_step(vxa, vya, pa, rho, mu, dl, dt, nu_extra=0.0):
    nx, ny = vxa.shape
    nvx, nvy, np_arr = vxa.copy(), vya.copy(), pa.copy()
    # nu_extra aumenta com AoA alto — evita boom
    nu_art = (mu / rho) + 0.012 + nu_extra
    max_v = 0.1
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dvx_dx = (vxa[i+1, j] - vxa[i-1, j]) / (2 * dl)
            dvx_dy = (vxa[i, j+1] - vxa[i, j-1]) / (2 * dl)

            nvx[i, j] -= (vxa[i, j] * dvx_dx + vya[i, j] * dvx_dy) * dt
            nvy[i, j] -= (vxa[i, j] * (vya[i+1, j] - vya[i-1, j]) / (2 * dl) + vya[i, j] * (vya[i, j+1] - vya[i, j-1]) / (2 * dl)) * dt

            diff_coef = nu_art * (dt / dl**2)
            # Clamp diffusion coefficient — evita explosão numérica em dt grande
            if diff_coef > 0.24: diff_coef = 0.24

            nvx[i, j] += diff_coef * (vxa[i+1, j] + vxa[i-1, j] + vxa[i, j+1] + vxa[i, j-1] - 4*vxa[i, j])
            nvy[i, j] += diff_coef * (vya[i+1, j] + vya[i-1, j] + vya[i, j+1] + vya[i, j-1] - 4*vya[i, j])

            div = (nvx[i+1, j] - nvx[i-1, j] + nvy[i, j+1] - nvy[i, j-1]) / (2 * dl)
            np_arr[i, j] = (pa[i+1, j] + pa[i-1, j] + pa[i, j+1] + pa[i, j-1]) / 4.0 - (rho * dl**2 / (4 * dt)) * div

            v_m = math.sqrt(nvx[i, j]**2 + nvy[i, j]**2)
            if v_m > max_v: max_v = v_m
    return nvx, nvy, np_arr, max_v

# ─────────────────────────────────────────────────────────
# MENU DARK
# ─────────────────────────────────────────────────────────

def launch_ui():
    root = tk.Tk()
    root.title("AeroSim v4.0.2")
    root.geometry("530x560")
    root.configure(bg="#0d1117")
    ACCENT, BG, PANEL, TEXT = "#58a6ff", "#0d1117", "#161b22", "#e6edf3"
    res = {"filepath": None, "aoa": 5.0, "velocity": 30.0, "altitude": 0.0, "save_gif": False, "cancelled": True}

    tk.Label(root, text="AEROSIM v4", bg=PANEL, fg=ACCENT, font=("Courier", 16, "bold"), pady=15).pack(fill="x")
    body = tk.Frame(root, bg=BG, padx=22, pady=10); body.pack(fill="both", expand=True)

    s1 = tk.LabelFrame(body, text=" GEOMETRIA (.DAT) ", bg=BG, fg=ACCENT, font=("Courier", 9, "bold"), bd=1, pady=10)
    s1.pack(fill="x", pady=8)
    path_v = tk.StringVar(value="(Selecione o arquivo)")
    tk.Entry(s1, textvariable=path_v, bg="#21262d", fg=TEXT, relief="flat").pack(side="left", fill="x", expand=True, padx=5)
    tk.Button(s1, text="DAT", command=lambda: path_v.set(filedialog.askopenfilename() or path_v.get())).pack(side="right", padx=5)

    s2 = tk.LabelFrame(body, text=" PARÂMETROS ", bg=BG, fg=ACCENT, font=("Courier", 9, "bold"), bd=1, pady=10)
    s2.pack(fill="x", pady=8)
    aoa_v, vel_v, alt_v = tk.DoubleVar(value=5.0), tk.DoubleVar(value=30.0), tk.DoubleVar(value=0.0)
    
    tk.Scale(s2, from_=-20, to=20, resolution=0.5, orient="horizontal", variable=aoa_v, label="AoA (Graus)", bg=BG, fg=TEXT, highlightthickness=0).pack(fill="x")
    tk.Scale(s2, from_=5, to=60, resolution=1, orient="horizontal", variable=vel_v, label="Velocidade (m/s)", bg=BG, fg=TEXT, highlightthickness=0).pack(fill="x")
    tk.Scale(s2, from_=0, to=5000, resolution=100, orient="horizontal", variable=alt_v, label="Altitude (m)", bg=BG, fg=TEXT, highlightthickness=0).pack(fill="x")

    gif_v = tk.BooleanVar(value=False)
    tk.Checkbutton(body, text="Salvar GIF ", variable=gif_v, bg=BG, fg=TEXT, selectcolor=PANEL).pack(pady=5)

    def run():
        res.update({"filepath": path_v.get() if os.path.isfile(path_v.get()) else None, 
                    "aoa": aoa_v.get(), "velocity": vel_v.get(), "altitude": alt_v.get(),
                    "save_gif": gif_v.get(), "cancelled": False})
        root.destroy()

    tk.Button(root, text="EXECUTAR SIMULAÇÃO", command=run, bg=ACCENT, fg=BG, font=("Courier", 12, "bold"), pady=12).pack(fill="x", padx=22, pady=15)
    root.mainloop(); return res

# ─────────────────────────────────────────────────────────
# MAIN (MAPEAMENTO GEOMETRIA 3.0 EXATO)
# ─────────────────────────────────────────────────────────

def main():
    cfg = launch_ui()
    if cfg["cancelled"]: return

    # Correção de Unpacking
    Ny, Nx = 360, 640
    v_target = cfg["velocity"]
    rho, mu = air_properties(cfg["altitude"])
    dl = 1.0 / Nx
    dt = 0.15 / (v_target + 1e-6)

    # Viscosidade extra escala com AoA — amortece separação severa
    aoa_deg = abs(cfg["aoa"])
    nu_extra = 0.0 + max(0.0, (aoa_deg - 8.0) * 0.004)  # zero até 8°, sobe depois

    # Carregamento robusto Geometria 3.0
    pts_raw = load_airfoil_robust(cfg["filepath"])
    label = os.path.splitext(os.path.basename(cfg["filepath"]))[0] if cfg["filepath"] else "Airfoil"

    # MAPEAMENTO DE COORDENADAS — Geometry Fix 5.0
    aoa = np.radians(cfg["aoa"])
    c, s = math.cos(aoa), math.sin(aoa)
    rot_matrix = np.array([[c, -s], [s, c]])

    # 1) Normaliza: perfil centrado em (0.5, 0) antes de rotar
    pts_norm = pts_raw.copy()
    pts_norm[:, 0] = pts_norm[:, 0] - 0.5   # centraliza X em 0
    pts_norm[:, 1] = pts_norm[:, 1]          # Y já centrado (~0)

    # 2) Rotaciona em torno do centro do perfil (0,0)
    pts_rot = (rot_matrix @ pts_norm.T).T

    # 3) Escala: corda ocupa 40% da largura do canvas — seguro pra qualquer AoA
    scale = Nx * 0.40

    # 4) Calcula bounding box PÓS-ROTAÇÃO e centraliza no canvas
    bx_min, bx_max = pts_rot[:, 0].min(), pts_rot[:, 0].max()
    by_min, by_max = pts_rot[:, 1].min(), pts_rot[:, 1].max()
    bx_center = (bx_min + bx_max) / 2.0
    by_center = (by_min + by_max) / 2.0

    pts_final = np.zeros_like(pts_rot)
    pts_final[:, 0] = (pts_rot[:, 0] - bx_center) * scale + Nx * 0.42
    # Inversão Y para OpenCV (Y cresce pra baixo)
    pts_final[:, 1] = Ny / 2.0 - (pts_rot[:, 1] - by_center) * scale

    # 5) Sanidade: se algum ponto saiu do canvas, escala inteira cabe dentro
    margin = 10
    x_out = max(0, -pts_final[:, 0].min() + margin, pts_final[:, 0].max() - (Nx - margin))
    y_out = max(0, -pts_final[:, 1].min() + margin, pts_final[:, 1].max() - (Ny - margin))
    if x_out > 0 or y_out > 0:
        # Encolhe escala para caber — mantém proporção
        shrink = min((Nx - 2*margin) / ((bx_max - bx_min) * scale + 1e-6),
                     (Ny - 2*margin) / ((by_max - by_min) * scale + 1e-6))
        scale *= shrink
        pts_final[:, 0] = (pts_rot[:, 0] - bx_center) * scale + Nx * 0.42
        pts_final[:, 1] = Ny / 2.0 - (pts_rot[:, 1] - by_center) * scale

    # Criação da Máscara (SEM ANTI-ALIASING para fidelidade geométrica)
    mask = np.zeros((Ny, Nx), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_final.astype(np.int32)], 255)
    objet = (mask.T > 127).astype(np.uint8)

    vx, vy, p = np.full((Nx, Ny), v_target, dtype=np.float64), np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    gif_frames = []

    try:
        for it in range(2500):
            vx, vy, p, v_max = solve_step(vx, vy, p, rho, mu, dl, dt, nu_extra)
            vx[1:-1, 1:-1] -= (dt/rho)*(p[2:, 1:-1]-p[:-2, 1:-1])/(2*dl)
            vy[1:-1, 1:-1] -= (dt/rho)*(p[1:-1, 2:]-p[1:-1, :-2])/(2*dl)

            # Clamp velocidade — anti-boom pós correção de pressão
            v_lim = v_target * 4.0
            np.clip(vx, -v_lim, v_lim, out=vx)
            np.clip(vy, -v_lim, v_lim, out=vy)

            dt = 0.35 * dl / max(v_max, v_target, 1.0)  # CFL conservador
            
            vx[objet==1], vy[objet==1] = 0, 0
            vx[0,:], vy[0,:] = v_target, 0

            if it % 40 == 0:
                mag = np.sqrt(vx**2 + vy**2).T
                display = cv2.merge([np.clip((mag/v_target)*127, 0, 255).astype(np.uint8)]*3)
                
                # Desenha geometria sólida (serrilhado original)
                display[mask > 0] = [40, 40, 40]
                
                cv2.imshow(f"AeroSim Pro v4.0.2 - {label}", display)
                if cfg["save_gif"]: gif_frames.append(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        if cfg["save_gif"] and gif_frames:
            imageio.mimsave(f"sim_{label}.gif", gif_frames, fps=24)

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()