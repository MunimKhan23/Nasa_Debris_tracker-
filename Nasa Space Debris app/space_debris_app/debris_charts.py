"""Matplotlib drawing functions (kept separate from the GUI so they can be tested headless)."""
from __future__ import annotations

import numpy as np

from debris_data import ALT_RANGES_KM, ORBITS, DebrisData

THEMES = {
    "dark": dict(bg="#0f0f23", panel="#1a1a2e", side="#16213e", fg="#e2e2e2", muted="#a0a0a0",
                 border="#2d2d4d", accent="#00d4ff", accent_text="#000000", input="#16213e",
                 button="#2d2d4d", hover="#3a3a60", success="#00ff88", warning="#ffdd57",
                 danger="#ff3860", globe="#0a0a1a", earth="#1E90FF", grid="#2d2d4d"),
    "light": dict(bg="#f8f9fa", panel="#ffffff", side="#f1f3f5", fg="#212529", muted="#6c757d",
                  border="#dee2e6", accent="#007bff", accent_text="#ffffff", input="#ffffff",
                  button="#e9ecef", hover="#d6dbe0", success="#28a745", warning="#e0a800",
                  danger="#dc3545", globe="#e9ecef", earth="#4aa3ff", grid="#dee2e6"),
}
PALETTE = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#8BC34A"]
RISK_COLORS_KEY = {"low": "success", "medium": "warning", "high": "danger"}


def _style_axes(ax, th):
    ax.set_facecolor(th["panel"])
    for s in ax.spines.values():
        s.set_color(th["border"])
    ax.tick_params(colors=th["fg"], labelsize=8)
    ax.yaxis.label.set_color(th["fg"])
    ax.xaxis.label.set_color(th["fg"])
    ax.grid(True, color=th["grid"], alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)


def _empty(fig, th, text="No objects match the current filters"):
    ax = fig.add_subplot(111)
    ax.set_facecolor(th["panel"])
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=th["muted"], transform=ax.transAxes)


def draw_orbit_chart(fig, data: DebrisData, mask, th):
    fig.clear()
    fig.set_facecolor(th["panel"])
    counts = data.count_by("orbit", mask)
    if sum(counts.values()) == 0:
        return _empty(fig, th)
    ax = fig.add_subplot(111)
    ax.set_facecolor(th["panel"])
    vals = list(counts.values())
    wedges, _, autot = ax.pie(vals, colors=PALETTE[:len(vals)], startangle=90, counterclock=False,
                              autopct=lambda p: f"{p:.0f}%" if p >= 4 else "", pctdistance=0.8,
                              wedgeprops=dict(width=0.42, edgecolor=th["panel"], linewidth=2),
                              textprops=dict(color="#000000", fontsize=8))
    ax.legend(wedges, [f"{k} ({v:,})" for k, v in counts.items()], loc="center left",
              bbox_to_anchor=(0.95, 0.5), frameon=False, labelcolor=th["fg"], fontsize=8)
    ax.text(0, 0, f"{sum(vals):,}\nobjects", ha="center", va="center", color=th["fg"], fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.7, top=0.97, bottom=0.03)


def draw_size_chart(fig, data: DebrisData, mask, th):
    fig.clear()
    fig.set_facecolor(th["panel"])
    counts = data.count_by("size_cat", mask)
    if sum(counts.values()) == 0:
        return _empty(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    labels = ["Small\n(<10 cm)", "Medium\n(10 cm–1 m)", "Large\n(>1 m)"]
    bars = ax.bar(labels, list(counts.values()), color=PALETTE[1:4], width=0.6)
    ax.bar_label(bars, color=th["fg"], fontsize=8, padding=2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Objects", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.2)


def draw_growth_chart(fig, data: DebrisData, mask, th):
    fig.clear()
    fig.set_facecolor(th["panel"])
    years, cum = data.growth(mask)
    if not cum or cum[-1] == 0:
        return _empty(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    ax.fill_between(years, cum, color=th["accent"], alpha=0.15)
    ax.plot(years, cum, color=th["accent"], linewidth=2.2, marker="o", markersize=5,
            markerfacecolor=th["panel"], markeredgewidth=2)
    for x, y in zip(years, cum):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                    color=th["fg"], fontsize=8)
    ax.set_xticks(years)
    ax.set_ylim(0, max(cum) * 1.15)
    ax.set_ylabel("Cumulative objects", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.12)


def draw_risk_chart(fig, data: DebrisData, mask, th):
    fig.clear()
    fig.set_facecolor(th["panel"])
    rb = data.risk_by_orbit(mask)
    if not any(sum(v) for v in rb.values()):
        return _empty(fig, th)
    ax = fig.add_subplot(111)
    _style_axes(ax, th)
    bottom = np.zeros(len(ORBITS))
    for risk, vals in rb.items():
        ax.bar(ORBITS, vals, bottom=bottom, label=risk.title(), color=th[RISK_COLORS_KEY[risk]], width=0.6)
        bottom += np.array(vals)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Objects", fontsize=8)
    ax.legend(frameon=False, labelcolor=th["fg"], fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.1)


# ------------------------------------------------------------------------------ 3D globe
def _radius(alt_km):
    """Map altitude (log scale) to a plotting radius: surface = 1, GEO ~ 1.9."""
    a = np.clip(np.asarray(alt_km, dtype=float), 200, 42000)
    return 1.1 + 0.8 * (np.log10(a) - np.log10(200)) / (np.log10(42000) - np.log10(200))


class Globe:
    """3D Earth with debris. Points hidden behind the planet are not drawn."""

    LIM = 1.9
    MAX_POINTS = 4000       # keeps rotation smooth even for huge catalogues

    def __init__(self, fig, data: DebrisData, th):
        self.fig, self.data, self.th = fig, data, th
        self.ax = None
        self._pts = []          # current scatter artists
        self._xyz = None        # (3, n) positions of the selected debris
        self._risk = None
        self._view = None

    def draw(self, mask, th=None, azim=30, elev=20):
        """Full redraw (new filters or new theme)."""
        d = self.data
        th = self.th = th or self.th
        fig = self.fig
        fig.clear()
        fig.set_facecolor(th["globe"])
        ax = self.ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        ax.set_facecolor(th["globe"])
        ax.set_axis_off()

        u, v = np.linspace(0, 2 * np.pi, 28), np.linspace(0, np.pi, 18)
        x, y, z = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=th["earth"], alpha=1.0, linewidth=0, antialiased=False, shade=True, zorder=1)
        ax.plot_wireframe(x, y, z, color="#ffffff", alpha=0.12, linewidth=0.4, rstride=3, cstride=3, zorder=1.1)
        t = np.linspace(0, 2 * np.pi, 120)
        for mid in (700, 20000, 35786):                       # LEO / MEO / GEO reference rings
            r = float(_radius(mid))
            ax.plot(r * np.cos(t), r * np.sin(t), 0 * t, color=th["muted"], alpha=0.45, linewidth=0.7, zorder=1.2)

        idx = np.flatnonzero(mask)
        if idx.size > self.MAX_POINTS:
            idx = idx[np.linspace(0, idx.size - 1, self.MAX_POINTS).astype(int)]
        lat, lon, r = np.radians(d.lat[idx]), np.radians(d.lon[idx]), _radius(d.altitude[idx])
        self._xyz = np.array([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])
        self._risk = d.risk[idx]
        self._pts = []

        lim = self.LIM
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.4)
        ax.view_init(elev=elev, azim=azim)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._view = None
        self.refresh()
        return ax

    def refresh(self, force=False):
        """Re-split visible / hidden points for the current camera angle. Returns True if redrawn."""
        ax, th = self.ax, self.th
        view = (round(ax.azim, 1), round(ax.elev, 1))
        if not force and view == self._view:
            return False
        self._view = view
        for p in self._pts:
            p.remove()
        self._pts = []
        az, el = np.radians(ax.azim), np.radians(ax.elev)
        cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        dot = cam @ self._xyz
        perp2 = (self._xyz ** 2).sum(axis=0) - dot ** 2
        visible = ~((dot < 0) & (perp2 < 1.0))               # not hidden by the Earth
        for risk, key in RISK_COLORS_KEY.items():
            sel = visible & (self._risk == risk)
            if sel.any():
                self._pts.append(ax.scatter(*self._xyz[:, sel], s=11 if risk == "high" else 7, c=th[key],
                                            depthshade=False, linewidths=0, zorder=2, label=f"{risk.title()} risk"))
        if self._pts:
            leg = ax.legend(handles=self._pts, loc="lower left", frameon=False, labelcolor=th["fg"], fontsize=7)
        return True

    def rotate(self, step=1.5):
        self.ax.view_init(elev=self.ax.elev, azim=self.ax.azim + step)
        self.refresh()

    def reset(self):
        self.ax.view_init(elev=20, azim=30)
        self.refresh()


def draw_globe(fig, data: DebrisData, mask, th, azim=30, elev=20):
    """Convenience wrapper (used by the tests)."""
    g = Globe(fig, data, th)
    g.draw(mask, th, azim, elev)
    return g
