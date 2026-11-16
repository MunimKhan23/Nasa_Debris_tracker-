"""Space Debris Visualizer - desktop app (Tkinter + Matplotlib).

Run:   python space_debris_app.py            (1,000 objects, reproducible)
       python space_debris_app.py --n 5000 --random
"""
from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import debris_charts as charts
from debris_ai import DebrisAI
from debris_data import COUNTRIES, ORBITS, DebrisData, format_size

PREFS = os.path.join(os.path.expanduser("~"), ".space_debris_app.json")

ORBIT_CHOICES = {"All orbits": None, "LEO – Low Earth": "LEO", "MEO – Medium Earth": "MEO",
                 "GEO – Geostationary": "GEO", "HEO – High / Elliptical": "HEO"}
SIZE_CHOICES = {"All sizes": None, "Small (1–10 cm)": "small", "Medium (10 cm – 1 m)": "medium",
                "Large (> 1 m)": "large"}
RISK_CHOICES = {"All risk levels": None, "Low": "low", "Medium": "medium", "High": "high"}

CHIPS = [("Largest in LEO", "What's the largest piece of debris in LEO?"),
         ("High-risk in GEO", "How many high-risk objects are in GEO?"),
         ("By size", "Show debris by size category"),
         ("Russia > 50 cm", "Russian debris larger than 50 cm since 2019"),
         ("Avg altitude China", "Average altitude of Chinese debris"),
         ("Near Earth 2020", "Show debris near Earth orbit from 2020"),
         ("Help", "help")]


# ============================================================================ widgets
class ChartPanel(ttk.Frame):
    """A card holding one Matplotlib chart."""

    def __init__(self, master, app, title, draw_fn):
        super().__init__(master, style="Card.TFrame", padding=10)
        self.app, self.draw_fn = app, draw_fn
        ttk.Label(self, text=title, style="CardHead.TLabel").pack(anchor="w")
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def redraw(self):
        th = self.app.theme
        self.canvas.get_tk_widget().configure(bg=th["panel"], highlightthickness=0)
        self.draw_fn(self.fig, self.app.data, self.app.mask, th)
        self.canvas.draw_idle()


class GlobePanel(ttk.Frame):
    """Rotating 3D globe (drag with the mouse to turn it)."""

    def __init__(self, master, app):
        super().__init__(master, style="Card.TFrame", padding=10)
        self.app = app
        self.rotating = True
        head = ttk.Frame(self, style="Card.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text="Orbital debris – drag to rotate", style="CardHead.TLabel").pack(side="left")
        ttk.Button(head, text="Reset view", command=self.reset).pack(side="right")
        self.btn = ttk.Button(head, text="Pause rotation", command=self.toggle)
        self.btn.pack(side="right", padx=6)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self.globe = charts.Globe(self.fig, app.data, app.theme)

    def redraw(self):
        th = self.app.theme
        self.canvas.get_tk_widget().configure(bg=th["globe"], highlightthickness=0)
        ax = self.globe.ax
        self.globe.draw(self.app.mask, th, azim=ax.azim if ax is not None else 30,
                        elev=ax.elev if ax is not None else 20)
        self.canvas.draw_idle()

    def toggle(self):
        self.rotating = not self.rotating
        self.btn.configure(text="Pause rotation" if self.rotating else "Resume rotation")

    def reset(self):
        self.globe.reset()
        self.canvas.draw_idle()

    def tick(self):
        """Called ~16x per second while the Overview tab is visible."""
        if self.globe.ax is None:
            return
        if self.rotating:
            self.globe.rotate(1.5)
            self.canvas.draw_idle()
        elif self.globe.refresh():                      # user dragged the globe
            self.canvas.draw_idle()


# ================================================================================ app
class App(tk.Tk):
    def __init__(self, n=1000, seed=42):
        super().__init__()
        self.title("Space Debris Visualizer")
        self.geometry("1320x860")
        self.minsize(1050, 680)

        self.data = DebrisData(n=n, seed=seed)
        self.ai = DebrisAI(self.data)
        self.mask = self.data.all_mask()
        self.theme_name = self._load_pref()
        self.theme = charts.THEMES[self.theme_name]
        self.active_tab = "overview"
        self.input_history: list[str] = []
        self.history_pos = 0
        self.ai_row_ids: list[int] = []
        self.trees: list[ttk.Treeview] = []

        base = tkfont.nametofont("TkDefaultFont")
        self.family = base.actual("family")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self._build_ui()
        self.apply_theme()
        self.refresh_all()
        self._welcome()
        self.show_tab("overview")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Alt-Key-1>", lambda e: self.show_tab("overview"))
        self.bind("<Alt-Key-2>", lambda e: self.show_tab("history"))
        self.bind("<Alt-Key-3>", lambda e: self.show_tab("statistics"))
        self.bind("<Alt-Key-4>", lambda e: self.show_tab("search"))
        self.bind("<Control-k>", lambda e: self.show_tab("search"))
        self.after(80, self._tick)

    # ----------------------------------------------------------------- preferences
    @staticmethod
    def _load_pref():
        try:
            with open(PREFS, encoding="utf-8") as fh:
                t = json.load(fh).get("theme", "dark")
                return t if t in charts.THEMES else "dark"
        except (OSError, ValueError):
            return "dark"

    def _save_pref(self):
        try:
            with open(PREFS, "w", encoding="utf-8") as fh:
                json.dump({"theme": self.theme_name}, fh)
        except OSError:
            pass

    def on_close(self):
        self._save_pref()
        self.destroy()

    # --------------------------------------------------------------------- layout
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        titles = ttk.Frame(header, style="Header.TFrame")
        titles.grid(row=0, column=0, sticky="w")
        ttk.Label(titles, text="Space Debris Visualizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Track, analyze & explore orbital debris", style="HeaderSub.TLabel").pack(anchor="w")
        self.theme_btn = ttk.Button(header, command=self.toggle_theme, width=11)
        self.theme_btn.grid(row=0, column=1, sticky="e")

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ---- sidebar
        side = ttk.Frame(body, style="Side.TFrame", padding=16, width=250)
        side.grid(row=0, column=0, sticky="ns")
        side.pack_propagate(False)
        self.nav = {}
        for key, label in (("overview", "Overview"), ("history", "History"),
                           ("statistics", "Statistics"), ("search", "AI Search")):
            b = ttk.Button(side, text=label, style="Nav.TButton", command=lambda k=key: self.show_tab(k))
            b.pack(fill="x", pady=2)
            self.nav[key] = b

        ttk.Label(side, text="Filters", style="SideHead.TLabel").pack(anchor="w", pady=(24, 6))
        self.filter_vars = {}
        self.combos = []
        year_choices = ["All years"] + [str(y) for y in self.data.years]
        country_choices = ["All countries"] + list(COUNTRIES)
        for key, label, values in (("orbit", "Orbit type", list(ORBIT_CHOICES)),
                                   ("size", "Size range", list(SIZE_CHOICES)),
                                   ("year", "Year added", year_choices),
                                   ("risk", "Risk level", list(RISK_CHOICES)),
                                   ("country", "Country", country_choices)):
            ttk.Label(side, text=label, style="Side.TLabel").pack(anchor="w", pady=(8, 2))
            var = tk.StringVar(value=values[0])
            cb = ttk.Combobox(side, textvariable=var, values=values, state="readonly")
            cb.pack(fill="x")
            cb.bind("<<ComboboxSelected>>", self._on_filter)
            self.filter_vars[key] = var
            self.combos.append(cb)
        ttk.Button(side, text="Reset filters", command=self.reset_filters).pack(fill="x", pady=(16, 0))
        ttk.Label(side, text="Filters apply to Overview, History and Statistics.\n"
                             "AI Search always searches the full catalogue.",
                  style="SideNote.TLabel", wraplength=215, justify="left").pack(anchor="w", pady=(14, 0))

        # ---- tabs (stacked in one grid cell, raised on demand)
        content = ttk.Frame(body, padding=14)
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self.tabs = {}
        for key in self.nav:
            f = ttk.Frame(content)
            f.grid(row=0, column=0, sticky="nsew")
            self.tabs[key] = f
        self._build_overview(self.tabs["overview"])
        self._build_history(self.tabs["history"])
        self._build_statistics(self.tabs["statistics"])
        self._build_search(self.tabs["search"])

        self.status = ttk.Label(self, text="", style="Status.TLabel", padding=(14, 4))
        self.status.grid(row=2, column=0, sticky="ew")

    def _card(self, parent, title):
        f = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10))
        ttk.Label(f, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w")
        value = ttk.Label(f, text="–", style="CardValue.TLabel")
        value.pack(anchor="w")
        sub = ttk.Label(f, text="", style="CardSub.TLabel")
        sub.pack(anchor="w")
        return f, value, sub

    def _build_overview(self, tab):
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        cards = ttk.Frame(tab)
        cards.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.cards = {}
        for i, (key, title) in enumerate((("total", "Objects shown"), ("high", "High-risk objects"),
                                          ("level", "Risk level"), ("latest", "Latest year"))):
            cards.columnconfigure(i, weight=1, uniform="cards")
            frame, value, sub = self._card(cards, title)
            frame.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            self.cards[key] = (frame, value, sub)

        self.globe_panel = GlobePanel(tab, self)
        self.globe_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        right = ttk.Frame(tab)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.orbit_panel = ChartPanel(right, self, "Debris by orbit type", charts.draw_orbit_chart)
        self.orbit_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.size_panel = ChartPanel(right, self, "Size distribution", charts.draw_size_chart)
        self.size_panel.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

    def _make_tree(self, parent, columns, widths, height=8, anchors=None):
        wrap = ttk.Frame(parent, style="Card.TFrame")
        tree = ttk.Treeview(wrap, columns=columns, show="headings", height=height, selectmode="browse")
        for i, c in enumerate(columns):
            tree.heading(c, text=c)
            tree.column(c, width=widths[i], minwidth=40, anchor=(anchors[i] if anchors else "w"), stretch=True)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.trees.append(tree)
        return wrap, tree

    def _build_history(self, tab):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        ttk.Label(tab, text="Debris history timeline", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        cols = ("Year", "Objects added", "LEO", "MEO", "GEO", "HEO", "High-risk", "Change vs prev. year")
        card = ttk.Frame(tab, style="Card.TFrame", padding=12)
        card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)
        ttk.Label(card, text="New objects added to the catalogue each year (current filters)",
                  style="CardSub.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        wrap, self.history_tree = self._make_tree(card, cols, (70, 110, 70, 70, 70, 70, 90, 150), height=12,
                                                  anchors=["center"] * 8)
        wrap.grid(row=1, column=0, sticky="nsew")

    def _build_statistics(self, tab):
        for c in (0, 1):
            tab.columnconfigure(c, weight=1, uniform="stat")
        tab.rowconfigure(0, weight=3)
        tab.rowconfigure(1, weight=2)
        self.growth_panel = ChartPanel(tab, self, "Debris growth over time", charts.draw_growth_chart)
        self.growth_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 12))
        self.risk_panel = ChartPanel(tab, self, "Risk level by orbit", charts.draw_risk_chart)
        self.risk_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 12))

        bottom = ttk.Frame(tab)
        bottom.grid(row=1, column=0, columnspan=2, sticky="nsew")
        for c in range(3):
            bottom.columnconfigure(c, weight=1, uniform="tbl")
        bottom.rowconfigure(0, weight=1)
        specs = (("Most crowded orbits", ("Orbit", "Objects", "Share"), (60, 70, 60), "orbit_tree"),
                 ("Largest debris objects", ("Name", "Size", "Orbit", "Risk"), (80, 70, 50, 60), "large_tree"),
                 ("Objects by country", ("Country", "Objects", "High-risk"), (80, 70, 70), "country_tree"))
        for i, (title, cols, widths, attr) in enumerate(specs):
            card = ttk.Frame(bottom, style="Card.TFrame", padding=10)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0 if i == 2 else 6))
            card.columnconfigure(0, weight=1)
            card.rowconfigure(1, weight=1)
            ttk.Label(card, text=title, style="CardHead.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
            wrap, tree = self._make_tree(card, cols, widths, height=6)
            wrap.grid(row=1, column=0, sticky="nsew")
            setattr(self, attr, tree)

    def _build_search(self, tab):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=2)
        tab.rowconfigure(4, weight=3)
        head = ttk.Frame(tab)
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="AI search agent", style="Section.TLabel").pack(anchor="w")
        ttk.Label(head, text="Ask about the debris in plain English – answers are computed exactly from the data.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))

        chat_card = ttk.Frame(tab, style="Card.TFrame", padding=8)
        chat_card.grid(row=1, column=0, sticky="nsew")
        chat_card.columnconfigure(0, weight=1)
        chat_card.rowconfigure(0, weight=1)
        self.chat = tk.Text(chat_card, wrap="word", relief="flat", borderwidth=0, padx=10, pady=8, height=8,
                            state="disabled", cursor="arrow")
        sb = ttk.Scrollbar(chat_card, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        self.chat.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        chips = ttk.Frame(tab)
        chips.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for label, query in CHIPS:
            ttk.Button(chips, text=label, style="Chip.TButton", command=lambda q=query: self.ask(q)).pack(
                side="left", padx=(0, 6))

        entry_row = ttk.Frame(tab)
        entry_row.grid(row=3, column=0, sticky="ew", pady=8)
        entry_row.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(entry_row)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=5)
        self.entry.bind("<Return>", lambda e: self.ask())
        self.entry.bind("<Up>", lambda e: self._recall(-1))
        self.entry.bind("<Down>", lambda e: self._recall(1))
        ttk.Button(entry_row, text="Send", style="Accent.TButton", command=self.ask).grid(
            row=0, column=1, padx=(8, 0))

        res = ttk.Frame(tab, style="Card.TFrame", padding=10)
        res.grid(row=4, column=0, sticky="nsew")
        res.columnconfigure(0, weight=1)
        res.rowconfigure(2, weight=1)
        self.result_title = ttk.Label(res, text="Results", style="CardHead.TLabel")
        self.result_title.grid(row=0, column=0, sticky="w")
        self.result_note = ttk.Label(res, text="Double-click a row for details.", style="CardSub.TLabel")
        self.result_note.grid(row=1, column=0, sticky="w", pady=(0, 6))
        wrap, self.result_tree = self._make_tree(res, ("Results",), (200,), height=8)
        wrap.grid(row=2, column=0, sticky="nsew")
        self.result_tree.bind("<Double-1>", self._on_result_double)

    # ---------------------------------------------------------------------- theme
    def apply_theme(self):
        th, s = self.theme, self.style
        self.configure(bg=th["bg"])
        fam = self.family
        s.configure(".", background=th["bg"], foreground=th["fg"], fieldbackground=th["input"],
                    bordercolor=th["border"], lightcolor=th["border"], darkcolor=th["border"],
                    troughcolor=th["bg"], focuscolor=th["accent"], insertcolor=th["fg"])
        for name, bg in (("TFrame", th["bg"]), ("Card.TFrame", th["panel"]), ("Side.TFrame", th["side"]),
                         ("Header.TFrame", th["panel"])):
            s.configure(name, background=bg)
        s.configure("TLabel", background=th["bg"], foreground=th["fg"])
        s.configure("Title.TLabel", background=th["panel"], font=(fam, 17, "bold"))
        s.configure("HeaderSub.TLabel", background=th["panel"], foreground=th["muted"])
        s.configure("Section.TLabel", font=(fam, 15, "bold"))
        s.configure("Muted.TLabel", foreground=th["muted"])
        s.configure("Status.TLabel", background=th["side"], foreground=th["muted"])
        s.configure("Side.TLabel", background=th["side"])
        s.configure("SideHead.TLabel", background=th["side"], font=(fam, 12, "bold"))
        s.configure("SideNote.TLabel", background=th["side"], foreground=th["muted"], font=(fam, 9))
        s.configure("CardHead.TLabel", background=th["panel"], font=(fam, 11, "bold"))
        s.configure("CardTitle.TLabel", background=th["panel"], foreground=th["muted"], font=(fam, 9, "bold"))
        s.configure("CardValue.TLabel", background=th["panel"], font=(fam, 22, "bold"))
        s.configure("CardSub.TLabel", background=th["panel"], foreground=th["muted"], font=(fam, 9))

        s.configure("TButton", background=th["button"], foreground=th["fg"], padding=(10, 6), borderwidth=1)
        s.map("TButton", background=[("active", th["hover"])])
        s.configure("Accent.TButton", background=th["accent"], foreground=th["accent_text"], padding=(16, 6))
        s.map("Accent.TButton", background=[("active", th["accent"])], foreground=[("active", th["accent_text"])])
        s.configure("Chip.TButton", padding=(8, 3), font=(fam, 9))
        s.configure("Nav.TButton", background=th["side"], foreground=th["fg"], anchor="w", padding=(12, 8),
                    borderwidth=0)
        s.map("Nav.TButton", background=[("active", th["hover"])])
        s.configure("NavActive.TButton", background=th["accent"], foreground=th["accent_text"], anchor="w",
                    padding=(12, 8), borderwidth=0)
        s.map("NavActive.TButton", background=[("active", th["accent"])], foreground=[("active", th["accent_text"])])

        s.configure("TCombobox", fieldbackground=th["input"], background=th["button"], foreground=th["fg"],
                    arrowcolor=th["fg"], bordercolor=th["border"], padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", th["input"])], foreground=[("readonly", th["fg"])],
              selectbackground=[("readonly", th["input"])], selectforeground=[("readonly", th["fg"])])
        s.configure("TEntry", fieldbackground=th["input"], foreground=th["fg"], padding=4)
        s.configure("Vertical.TScrollbar", background=th["button"], troughcolor=th["panel"],
                    arrowcolor=th["fg"], bordercolor=th["panel"])
        s.configure("Treeview", background=th["panel"], fieldbackground=th["panel"], foreground=th["fg"],
                    rowheight=26, borderwidth=0)
        s.configure("Treeview.Heading", background=th["button"], foreground=th["fg"], relief="flat",
                    font=(fam, 9, "bold"), padding=4)
        s.map("Treeview", background=[("selected", th["accent"])], foreground=[("selected", th["accent_text"])])
        s.map("Treeview.Heading", background=[("active", th["hover"])])

        self.option_add("*TCombobox*Listbox.background", th["input"])
        self.option_add("*TCombobox*Listbox.foreground", th["fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", th["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", th["accent_text"])
        for cb in getattr(self, "combos", []):             # drop cached pop-downs so they pick up new colours
            try:
                cb.tk.call("destroy", str(cb.tk.call("ttk::combobox::PopdownWindow", cb)))
            except tk.TclError:
                pass

        for tree in self.trees:
            tree.tag_configure("high", foreground=th["danger"])
            tree.tag_configure("medium", foreground=th["warning"])
            tree.tag_configure("low", foreground=th["success"])
        self.chat.configure(bg=th["bg"], fg=th["fg"], insertbackground=th["fg"], selectbackground=th["accent"])
        self.chat.tag_configure("you", foreground=th["accent"], font=(fam, 10, "bold"))
        self.chat.tag_configure("ai", foreground="#9b8cff" if self.theme_name == "dark" else "#6f42c1",
                                font=(fam, 10, "bold"))
        self.chat.tag_configure("body", foreground=th["fg"], font=(fam, 10), spacing3=2)
        self.chat.tag_configure("meta", foreground=th["muted"], font=(fam, 8), spacing3=8)
        self.theme_btn.configure(text="☀ Light" if self.theme_name == "dark" else "☾ Dark")
        for key, btn in self.nav.items():
            btn.configure(style="NavActive.TButton" if key == self.active_tab else "Nav.TButton")

    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.theme = charts.THEMES[self.theme_name]
        self.apply_theme()
        self.refresh_all()
        self._save_pref()

    # -------------------------------------------------------------------- filters
    def _on_filter(self, event=None):
        if event is not None:
            event.widget.selection_clear()
        self.refresh_all()

    def reset_filters(self):
        for var, cb in zip(self.filter_vars.values(), self.combos):
            var.set(cb["values"][0])
        self.refresh_all()

    def current_mask(self):
        v = self.filter_vars
        year = v["year"].get()
        country = v["country"].get()
        return self.data.mask(orbit=ORBIT_CHOICES[v["orbit"].get()], size_cat=SIZE_CHOICES[v["size"].get()],
                              year=None if year.startswith("All") else int(year),
                              risk=RISK_CHOICES[v["risk"].get()],
                              country=None if country.startswith("All") else country)

    def show_tab(self, name):
        self.active_tab = name
        self.tabs[name].tkraise()
        for key, btn in self.nav.items():
            btn.configure(style="NavActive.TButton" if key == name else "Nav.TButton")
        if name == "search":
            self.entry.focus_set()

    # ------------------------------------------------------------------- refreshing
    def refresh_all(self):
        d = self.data
        self.mask = self.current_mask()
        self._update_cards()
        for p in (self.orbit_panel, self.size_panel, self.growth_panel, self.risk_panel, self.globe_panel):
            p.redraw()
        self._fill_history()
        self._fill_stats()
        n = int(self.mask.sum())
        self.status.configure(text=f"Showing {n:,} of {len(d):,} objects   |   "
                                   f"AI search covers all {len(d):,} objects")

    def _update_cards(self):
        d, m, th = self.data, self.mask, self.theme
        n, total = int(m.sum()), len(self.data)
        high = int((m & (d.risk == "high")).sum())
        level = d.risk_level(high, n)
        latest, prev = d.years[-1], d.years[-2] if len(d.years) > 1 else None
        c_latest = int((m & (d.year == latest)).sum())
        c_prev = int((m & (d.year == prev)).sum()) if prev else 0
        change = f"{(c_latest - c_prev) / c_prev * 100:+.0f}% vs {prev}" if c_prev else "no earlier year to compare"
        level_color = {"High": th["danger"], "Medium": th["warning"], "Low": th["success"]}.get(level, th["fg"])
        data = {"total": (f"{n:,}", f"{n / total * 100:.0f}% of the catalogue", th["fg"]),
                "high": (f"{high:,}", f"{high / n * 100:.1f}% of selection" if n else "–", th["fg"]),
                "level": (level, "from share of high-risk objects", level_color),
                "latest": (f"{c_latest:,}", change, th["fg"])}
        for key, (value, sub, color) in data.items():
            _, vlabel, slabel = self.cards[key]
            vlabel.configure(text=value, foreground=color)
            slabel.configure(text=sub)
        self.cards["latest"][0].winfo_children()[0].configure(text=f"ADDED IN {latest}")

    def _fill_history(self):
        d, m = self.data, self.mask
        tree = self.history_tree
        tree.delete(*tree.get_children())
        rows, prev = [], None
        for y in d.years:
            ym = m & (d.year == y)
            tot = int(ym.sum())
            per = [int((ym & (d.orbit == o)).sum()) for o in ORBITS]
            high = int((ym & (d.risk == "high")).sum())
            change = f"{(tot - prev) / prev * 100:+.0f}%" if prev else "—"
            rows.append((str(y), f"{tot:,}", *[f"{p:,}" for p in per], f"{high:,}", change))
            prev = tot
        for r in reversed(rows):
            tree.insert("", "end", values=r)

    def _fill_stats(self):
        d, m = self.data, self.mask
        n = max(int(m.sum()), 1)
        for t in (self.orbit_tree, self.large_tree, self.country_tree):
            t.delete(*t.get_children())
        for orbit, c in sorted(d.count_by("orbit", m).items(), key=lambda kv: -kv[1]):
            self.orbit_tree.insert("", "end", values=(orbit, f"{c:,}", f"{c / n * 100:.1f}%"))
        for i in d.largest(m, 8):
            self.large_tree.insert("", "end", values=(d.name[i], format_size(d.size_cm[i]), d.orbit[i],
                                                      str(d.risk[i]).title()), tags=(str(d.risk[i]),))
        for country, c in sorted(d.count_by("country", m).items(), key=lambda kv: -kv[1]):
            high = int((m & (d.country == country) & (d.risk == "high")).sum())
            self.country_tree.insert("", "end", values=(country, f"{c:,}", f"{high:,}"))

    def _tick(self):
        if self.active_tab == "overview":
            self.globe_panel.tick()
        self.after(60, self._tick)

    # ---------------------------------------------------------------- AI search tab
    def _chat_add(self, who, text, meta=None):
        self.chat.configure(state="normal")
        self.chat.insert("end", ("You" if who == "you" else "Assistant") + "\n", who)
        self.chat.insert("end", text + "\n", "body")
        if meta:
            self.chat.insert("end", meta + "\n", "meta")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _welcome(self):
        self._chat_add("ai", "Hello! I'm your Space Debris Assistant. Ask me things like:\n"
                             "• “Show debris near Earth orbit from 2020”\n"
                             "• “What's the largest piece of debris in LEO?”\n"
                             "• “How many high-risk objects are in GEO?”\n"
                             "Type “help” for everything I understand, or click an example below.")

    def _recall(self, step):
        if not self.input_history:
            return "break"
        self.history_pos = max(0, min(len(self.input_history), self.history_pos + step))
        self.entry.delete(0, "end")
        if self.history_pos < len(self.input_history):
            self.entry.insert(0, self.input_history[self.history_pos])
        return "break"

    def ask(self, query=None):
        text = (query if query is not None else self.entry.get()).strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.input_history.append(text)
        self.history_pos = len(self.input_history)
        self.show_tab("search")
        resp = self.ai.ask(text)
        self._chat_add("you", text)
        meta = (resp.interpretation + "   ·   " if resp.interpretation else "") + f"{resp.elapsed_ms:.2f} ms"
        self._chat_add("ai", resp.message, meta)

        tree = self.result_tree
        tree.delete(*tree.get_children())
        self.ai_row_ids = list(resp.record_ids)
        if resp.columns:
            tree.configure(columns=resp.columns)
            widths = {"ID": 90, "Size": 90, "Orbit": 70, "Year": 70, "Risk": 80}
            for c in resp.columns:
                tree.heading(c, text=c)
                tree.column(c, width=widths.get(c, 110), anchor="w", stretch=True)
            for row in resp.rows:
                tag = row[-1].lower() if resp.kind == "list" else ""
                tree.insert("", "end", values=row, tags=(tag,) if tag else ())
            self.result_title.configure(text=f"Results – {resp.total:,} matching object"
                                             f"{'' if resp.total == 1 else 's'}")
            self.result_note.configure(text="Double-click a row for details." if resp.kind == "list" else "")
        else:
            self.result_title.configure(text="Results")
            self.result_note.configure(text="This answer has no table – see the chat above.")

    def _on_result_double(self, event):
        sel = self.result_tree.selection()
        if not sel or not self.ai_row_ids:
            return
        pos = self.result_tree.index(sel[0])
        if pos < len(self.ai_row_ids):
            self._detail(self.ai_row_ids[pos])

    def _detail(self, debris_id):
        import numpy as np
        d = self.data
        rec = d.record(int(np.flatnonzero(d.id == debris_id)[0]))
        win = tk.Toplevel(self)
        win.title(rec["name"])
        win.configure(bg=self.theme["panel"])
        win.transient(self)
        win.resizable(False, False)
        frame = ttk.Frame(win, style="Card.TFrame", padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=rec["name"], style="CardValue.TLabel").grid(row=0, column=0, columnspan=2,
                                                                          sticky="w", pady=(0, 10))
        fields = (("Orbit", rec["orbit"]), ("Altitude", f"{rec['altitude']:,} km"),
                  ("Size", f"{format_size(rec['size_cm'])}  ({rec['size_cat']})"),
                  ("Country", rec["country"]), ("Year added", rec["year"]), ("Risk level", rec["risk"].title()),
                  ("Latitude", f"{rec['lat']:.2f}°"), ("Longitude", f"{rec['lon']:.2f}°"))
        for r, (k, v) in enumerate(fields, start=1):
            ttk.Label(frame, text=k, style="CardSub.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 24), pady=3)
            color = {"high": self.theme["danger"], "medium": self.theme["warning"],
                     "low": self.theme["success"]}.get(str(v).lower()) if k == "Risk level" else None
            lab = ttk.Label(frame, text=str(v), style="CardHead.TLabel")
            if color:
                lab.configure(foreground=color)
            lab.grid(row=r, column=1, sticky="w", pady=3)
        ttk.Button(frame, text="Close", style="Accent.TButton", command=win.destroy).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="e", pady=(16, 0))
        win.bind("<Escape>", lambda e: win.destroy())


def main():
    ap = argparse.ArgumentParser(description="Space Debris Visualizer")
    ap.add_argument("--n", type=int, default=1000, help="number of debris objects (default 1000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--random", action="store_true", help="different data on every launch")
    args = ap.parse_args()
    try:                                                   # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App(n=args.n, seed=None if args.random else args.seed).mainloop()


if __name__ == "__main__":
    main()
