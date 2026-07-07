# ================================================================= #
# theme.py — Dark Wing CFD color palette, matplotlib rcParams, cmaps#
# ================================================================= #

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Crystal Dark Palette ─────────────────────────────────────────────
BG           = "#040814"
PANEL        = "#2a3442"
ACCENT       = "#45515c"
TEXT         = "#f3ecec"
MUTED        = "#0933f0"
GREEN        = "#035afa"
YELLOW       = "#ffe066"
RED          = "#ff3b3b"
BORDER       = "#1a3a5c"
CRYSTAL_HIGH = "#f5f5f5"
CRYSTAL_MID  = "#000305"
CRYSTAL_DIM  = "#000408"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": BG,
    "axes.edgecolor":  ACCENT, "axes.labelcolor": MUTED,
    "axes.titlecolor": ACCENT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": BORDER, "grid.linewidth": 0.4, "text.color": TEXT,
    "font.family": "monospace", "font.size": 8,
})

_cmap = LinearSegmentedColormap.from_list("pp", [
    "#040814","#051830","#0a3060","#0d5080","#00a0c8",
    "#00e5ff","#a0f0ff","#ffffff"], N=512)
try:
    matplotlib.colormaps.register(_cmap)
except ValueError:
    pass

_schlieren_gray_cmap = LinearSegmentedColormap.from_list("schlieren_gray", [
    "#050505","#1a1a1a","#333333","#4d4d4d","#666666",
    "#808080","#999999","#b3b3b3","#cccccc","#e6e6e6","#ffffff"], N=512)
try:
    matplotlib.colormaps.register(_schlieren_gray_cmap)
except ValueError:
    pass

CMAPS = {
    "turbo":          "turbo",
    "viridis":        "viridis",
    "inferno":        "inferno",
    "coolwarm":       "coolwarm",
    "schlieren_gray": "schlieren_gray",
}

