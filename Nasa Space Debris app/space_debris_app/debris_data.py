"""Space debris dataset: generation, filtering and statistics.

All records live in NumPy arrays so every filter / statistic is a vectorised
operation (1,000 rows or 1,000,000 rows - it stays fast).
"""
from __future__ import annotations

import numpy as np

ORBITS = ("LEO", "MEO", "GEO", "HEO")
SIZE_CATS = ("small", "medium", "large")
COUNTRIES = ("USA", "Russia", "China", "ESA", "India", "Japan", "Other")
RISKS = ("low", "medium", "high")
YEAR_MIN, YEAR_MAX = 2018, 2022

# Size categories (all sizes are stored in centimetres)
SIZE_RANGES_CM = {"small": (1.0, 10.0), "medium": (10.0, 100.0), "large": (100.0, 1000.0)}
# Altitude ranges in km
ALT_RANGES_KM = {
    "LEO": (200, 1200),
    "MEO": (2000, 35000),
    "GEO": (35586, 36286),
    "HEO": (1000, 41000),
}


def format_size(cm: float) -> str:
    """1 - 99 cm are shown in cm, anything bigger in metres."""
    cm = float(cm)
    return f"{cm:.1f} cm" if cm < 100 else f"{cm / 100:.2f} m"


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (str, int, np.integer)):
        return None if str(value).lower() == "all" else [value]
    return list(value)


class DebrisData:
    """Container for the debris catalogue."""

    def __init__(self, n: int = 1000, seed: int | None = 42):
        rng = np.random.default_rng(seed)
        self.n = n
        self.id = np.arange(1, n + 1)
        self.name = np.array([f"DEB-{i:04d}" for i in self.id])
        self.year = rng.integers(YEAR_MIN, YEAR_MAX + 1, n)
        self.size_cat = rng.choice(SIZE_CATS, n)
        self.orbit = rng.choice(ORBITS, n)
        self.country = rng.choice(COUNTRIES, n)
        self.risk = rng.choice(RISKS, n, p=[0.4, 0.3, 0.3])

        lo = np.array([SIZE_RANGES_CM[c][0] for c in self.size_cat])
        hi = np.array([SIZE_RANGES_CM[c][1] for c in self.size_cat])
        self.size_cm = np.round(rng.uniform(lo, hi), 1)

        lo = np.array([ALT_RANGES_KM[o][0] for o in self.orbit])
        hi = np.array([ALT_RANGES_KM[o][1] for o in self.orbit])
        self.altitude = rng.integers(lo, hi + 1)

        # uniform points on a sphere
        self.lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        self.lon = rng.uniform(-180, 180, n)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return self.n

    @property
    def years(self) -> list[int]:
        return sorted(int(y) for y in np.unique(self.year))

    def all_mask(self) -> np.ndarray:
        return np.ones(self.n, dtype=bool)

    def record(self, i: int) -> dict:
        """Row `i` (0-based array index) as a plain dict."""
        return {
            "id": int(self.id[i]),
            "name": str(self.name[i]),
            "orbit": str(self.orbit[i]),
            "size_cm": float(self.size_cm[i]),
            "size_cat": str(self.size_cat[i]),
            "altitude": int(self.altitude[i]),
            "country": str(self.country[i]),
            "year": int(self.year[i]),
            "risk": str(self.risk[i]),
            "lat": float(self.lat[i]),
            "lon": float(self.lon[i]),
        }

    # ----------------------------------------------------------------- filters
    def mask(self, orbit=None, size_cat=None, year=None, risk=None, country=None) -> np.ndarray:
        """Boolean mask. Every argument may be None/'all', a value or a list."""
        m = self.all_mask()
        for arr, val in ((self.orbit, orbit), (self.size_cat, size_cat), (self.risk, risk),
                         (self.country, country)):
            vals = _as_list(val)
            if vals:
                m &= np.isin(arr, vals)
        yrs = _as_list(year)
        if yrs:
            m &= np.isin(self.year, [int(y) for y in yrs])
        return m

    # -------------------------------------------------------------- statistics
    def count_by(self, field: str, mask: np.ndarray | None = None) -> dict:
        """Counts per category of orbit / size_cat / country / risk / year."""
        mask = self.all_mask() if mask is None else mask
        arr = getattr(self, field)
        cats = {"orbit": ORBITS, "size_cat": SIZE_CATS, "country": COUNTRIES,
                "risk": RISKS, "year": self.years}[field]
        sel = arr[mask]
        return {c: int(np.count_nonzero(sel == c)) for c in cats}

    def growth(self, mask: np.ndarray | None = None) -> tuple[list[int], list[int]]:
        """(years, cumulative number of objects) for the selection."""
        per_year = self.count_by("year", mask)
        years = list(per_year)
        return years, [int(v) for v in np.cumsum(list(per_year.values()))]

    def risk_by_orbit(self, mask: np.ndarray | None = None) -> dict:
        mask = self.all_mask() if mask is None else mask
        return {r: [int(np.count_nonzero(mask & (self.orbit == o) & (self.risk == r)))
                    for o in ORBITS] for r in RISKS}

    @staticmethod
    def risk_level(high: int, total: int) -> str:
        """Overall risk level from the share of high-risk objects."""
        if total == 0:
            return "n/a"
        share = high / total
        return "High" if share > 0.40 else "Medium" if share > 0.25 else "Low"

    def largest(self, mask: np.ndarray | None = None, k: int = 5) -> np.ndarray:
        """Array indices of the k largest objects in the selection (ties -> lower id)."""
        idx = np.flatnonzero(self.all_mask() if mask is None else mask)
        order = np.lexsort((self.id[idx], -self.size_cm[idx]))
        return idx[order][:k]
