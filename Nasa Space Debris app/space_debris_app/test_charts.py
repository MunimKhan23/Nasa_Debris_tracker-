"""Render every chart headless to PNGs (checks the drawing code without a display)."""
import sys
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import debris_charts as c
from debris_data import DebrisData

out = sys.argv[1] if len(sys.argv) > 1 else "."
d = DebrisData()
for theme in ("dark", "light"):
    th = c.THEMES[theme]
    for name, fn in (("orbit", c.draw_orbit_chart), ("size", c.draw_size_chart),
                     ("growth", c.draw_growth_chart), ("risk", c.draw_risk_chart), ("globe", c.draw_globe)):
        fig = Figure(figsize=(6, 4), dpi=90)
        fn(fig, d, d.all_mask(), th)
        fig.savefig(f"{out}/{theme}_{name}.png", facecolor=fig.get_facecolor())
    # empty selection must not crash
    for fn in (c.draw_orbit_chart, c.draw_size_chart, c.draw_growth_chart, c.draw_risk_chart, c.draw_globe):
        fig = Figure(figsize=(4, 3))
        fn(fig, d, d.mask(year=1999), th)

# rotation cost: how long does one globe frame take?
fig = Figure(figsize=(6, 4), dpi=90)
from matplotlib.backends.backend_agg import FigureCanvasAgg
FigureCanvasAgg(fig)
g = c.draw_globe(fig, d, d.all_mask(), c.THEMES["dark"])
t = time.perf_counter()
for i in range(10):
    g.rotate(3)
    fig.canvas.draw()
print(f"globe frame: {(time.perf_counter() - t) / 10 * 1000:.0f} ms")
print("charts OK")
