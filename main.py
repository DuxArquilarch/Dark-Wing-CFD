
import sys
import os
import time
import math
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

DEBUG = True
def dbg(msg):
    if DEBUG:
        print(f"[DarkWingCFD][{time.strftime('%H:%M:%S')}] {msg}", flush=True)


LOGO_FILENAME = "pfp.png"


def resource_path(filename):
    """Resolve a bundled asset path that works both when running from
    source and when frozen into a single-file executable (PyInstaller)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)


class LinearLoadingBar:
    """horizontal progress bar."""
    def __init__(self, master, bg, panel, accent, border, width=460, height=22):
        self.width  = width
        self.height = height
        self.panel  = panel
        self.accent = accent
        self.border = border

        self.canvas = tk.Canvas(master, width=width, height=height, bg=bg, highlightthickness=0)
        self.canvas.pack(pady=(0, 10))

        self.canvas.create_rectangle(1, 1, width - 1, height - 1,
                                      fill=panel, outline=border, width=1)
        self.fill_rect = self.canvas.create_rectangle(2, 2, 2, height - 2,
                                                        fill=accent, outline="")
        self.pct_text = self.canvas.create_text(width / 2, height / 2, text="0%",
                                                  fill=accent, font=("Courier", 9, "bold"))

    def set_progress(self, percentage):
        percentage = max(0, min(100, percentage))
        w = 2 + (self.width - 4) * (percentage / 100.0)
        self.canvas.coords(self.fill_rect, 2, 2, w, self.height - 2)
        self.canvas.itemconfig(self.pct_text, text=f"{int(percentage)}%")
        self.canvas.update_idletasks()


def load_logo_image(size, filename=LOGO_FILENAME):
    path = resource_path(filename)
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        dbg(f"could not load logo image '{path}': {e}")
        return None


def build_loading_screen(root, BG, PANEL, ACCENT, BORDER, MUTED):
    loading_frame = tk.Frame(root, bg=BG)
    loading_frame.place(relx=0.5, rely=0.5, anchor="center")

    logo_size = 320
    logo_canvas = tk.Canvas(loading_frame, width=logo_size, height=logo_size,
                             bg=BG, highlightthickness=0)
    logo_canvas.pack(pady=(0, 6))

    logo_image = load_logo_image(logo_size)
    if logo_image is not None:
        # Keep a reference on the canvas itself so it isn't garbage
        # collected once this function returns.
        logo_canvas.image = logo_image
        logo_canvas.create_image(logo_size / 2, logo_size / 2, image=logo_image)

    tk.Label(loading_frame, text="◈ DARK WING CFD ◈", fg=ACCENT, bg=BG,
             font=("Helvetica", 16, "bold")).pack(pady=(0, 10))

    bar = LinearLoadingBar(loading_frame, BG, PANEL, ACCENT, BORDER, width=460, height=22)

    loading_label = tk.Label(loading_frame, text="INICIALIZANDO SISTEMA...",
                              fg=ACCENT, bg=BG, font=("Helvetica", 12, "bold"))
    loading_label.pack(pady=(0, 4))

    root.update()
    return loading_frame, loading_label, bar


def main():
    dbg("starting")
    root = tk.Tk()
    dbg("tk window created")
    root.withdraw()

    from theme import BG, PANEL, ACCENT, BORDER, MUTED
    dbg("theme loaded")

    root.title("► Dark Wing CFD ◄")
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")

    w, h = 1440, 900
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.deiconify()

    loading_frame, loading_label, bar = build_loading_screen(root, BG, PANEL, ACCENT, BORDER, MUTED)

    def update_progress(value, text):
        loading_label.config(text=text)
        bar.set_progress(value)
        root.update()
        dbg(f"progress {value:3d}%  {text}")

    # Cada etapa do progresso corresponde a um import/estágio real de
    # inicialização — a barra só avança quando o trabalho correspondente
    # de fato terminou (sem sleeps artificiais).
    total_steps = 8
    step = [0]
    def tick(text):
        step[0] += 1
        update_progress(int(step[0] / total_steps * 100), text)

    tick("⚙ Carregando geometria...")
    import geometry  # noqa: F401
    dbg("geometry module ready")

    tick("► Carregando presets de asa ◄")
    import wing_presets  # noqa: F401
    dbg("wing_presets module ready")

    tick("► Inicializando Euler Solver ◄")
    import flow_field  # noqa: F401
    dbg("flow_field module ready")

    tick("► Inicializando Navier-Stokes ◄")
    import ns_solver  # noqa: F401
    dbg("ns_solver module ready")

    tick("► Carregando janela N-S e panel method ◄")
    import ns_window  # noqa: F401
    import panel_method  # noqa: F401
    dbg("ns_window / panel_method modules ready")

    tick("► Carregando shaders e malhas (GUI) ◄")
    from gui import CFDVisualizer
    dbg("gui module imported")

    tick("► Construindo interface principal ◄")
    visualizer = CFDVisualizer(root)
    root.update()
    dbg("CFDVisualizer fully constructed")

    tick("► Dark Wing CFD Pronto! ◄")

    loading_frame.destroy()
    dbg("loading frame destroyed")

    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()

    dbg("main window visible, entering mainloop")
    root.mainloop()
    dbg("window closed")


def _enable_ansi_on_windows():
    if os.name == "nt":
        try:
            os.system("")  # enables ANSI escape processing in cmd.exe / Windows Terminal
        except Exception:
            pass

RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def print_crash_report(exc):
    _enable_ansi_on_windows()
    print(f"\n{RED}Traceback (most recent call last):{RESET}", flush=True)
    traceback.print_exc()
    msg = f"{type(exc).__name__}: {exc}"
    print(f"\n{RED}🔴 🔴 Error: {msg} 🔴 🔴{RESET}", flush=True)
    print(f"{DIM}[DarkWingCFD] fatal — see traceback above{RESET}", flush=True)
    print(f"{RED}🥀 🥀 Error: Faaaaaah 🥀 🥀{RESET}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_crash_report(exc)
        input("\n[ERRO] O programa falhou. Pressione ENTER...")
