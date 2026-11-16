"""Natural-language search agent for the debris catalogue.

A deterministic, rule-based parser: every question is turned into exact NumPy
filters, so the numbers it reports are always the real answer for the data
(no guessing, no random sample rows) and a query takes well under a millisecond.

Understands, in any combination:
  orbit      LEO / low earth / near earth, MEO, GEO / geostationary, HEO
  size       large / medium / small, "larger than 50 cm", "under 2 m"
  altitude   "above 1000 km", "between 500 and 800 km"
  year       "2020", "since 2019", "before 2021", "between 2019 and 2021"
  risk       high risk / dangerous, medium / moderate, low / safe
  country    USA, Russia, China, ESA/Europe, India, Japan, other countries
  questions  how many ...  |  average/median altitude/size ...
             largest / smallest / highest / lowest / newest / oldest ...
             ... by orbit / size / country / year / risk   (breakdowns)
  lookups    "DEB-0042"
  limits     "top 5", "show 20", "show all"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from debris_data import DebrisData, format_size

EXAMPLES = [
    "Show debris near Earth orbit from 2020",
    "What's the largest piece of debris in LEO?",
    "How many high-risk objects are in GEO?",
    "Show debris by size category",
    "Russian debris larger than 50 cm since 2019",
    "Average altitude of Chinese debris",
]

ROW_COLUMNS = ("ID", "Orbit", "Size", "Altitude (km)", "Country", "Year", "Risk")


@dataclass
class AIResponse:
    message: str
    kind: str = "info"                 # list | count | average | breakdown | greeting | help | none | error
    columns: tuple = ()                # result-table header
    rows: list = field(default_factory=list)          # result-table rows (tuples of str)
    record_ids: list = field(default_factory=list)    # debris ids aligned with rows (list tables only)
    total: int = 0                     # number of matching objects
    interpretation: str = ""           # how the question was understood
    elapsed_ms: float = 0.0


# --------------------------------------------------------------------- patterns
ORBIT_ALIASES = [
    ("LEO", [r"\bleo\b", r"\blow earth(?: orbits?)?\b", r"\bnear earth(?: orbits?)?\b"]),
    ("MEO", [r"\bmeo\b", r"\b(?:medium|mid) earth(?: orbits?)?\b"]),
    ("GEO", [r"\bgeo\b", r"\bgeostationary(?: orbits?)?\b", r"\bgeosynchronous(?: orbits?)?\b"]),
    ("HEO", [r"\bheo\b", r"\bhigh earth(?: orbits?)?\b", r"\bhighly elliptical(?: orbits?)?\b",
             r"\belliptical(?: orbits?)?\b"]),
]
COUNTRY_ALIASES = [
    ("USA", [r"\busa\b", r"\bunited states\b", r"\bamerica\w*\b"]),
    ("Russia", [r"\brussia\w*\b", r"\bsoviet\b", r"\bussr\b"]),
    ("China", [r"\bchina\b", r"\bchinese\b"]),
    ("ESA", [r"\besa\b", r"\beurope\w*\b"]),
    ("India", [r"\bindia\w*\b"]),
    ("Japan", [r"\bjapan\w*\b"]),
    ("Other", [r"\bother countr(?:y|ies)\b", r"\bother nations?\b"]),
]
RISK_ALIASES = [
    ("high", [r"\bhigh(?:est)?[ -]?risk\w*", r"\bdangerous\b", r"\bhazard\w*", r"\bcritical\b",
              r"\briskiest\b", r"\brisky\b", r"\bmost dangerous\b"]),
    ("medium", [r"\b(?:medium|moderate|mid)[ -]?risk\w*", r"\bmoderate\b"]),
    ("low", [r"\blow(?:est)?[ -]?risk\w*", r"\bsafe\w*", r"\bharmless\b"]),
]

NUM = r"\d+(?:\.\d+)?"
UNIT = r"(?:km|kilomet(?:er|re)s?|cm|centimet(?:er|re)s?|mm|millimet(?:er|re)s?|met(?:er|re)s?|m)"
OP_WORDS = {
    ">": r"larger than|bigger than|greater than|more than|longer than|higher than|above|over|exceeding|beyond",
    ">=": r"at least|no less than|not less than|minimum of",
    "<": r"smaller than|less than|lower than|shorter than|under|below|beneath",
    "<=": r"at most|no more than|not more than|up to|maximum of|within",
}
RE_BETWEEN = re.compile(rf"\b(?:between|from)\s+({NUM})\s*({UNIT})?\s+(?:and|to)\s+({NUM})\s*({UNIT})\b")
RE_CMP = re.compile(rf"\b({'|'.join(f'(?:{v})' for v in OP_WORDS.values())})\s+(?:of\s+)?({NUM})\s*({UNIT})\b")

RE_YEAR_RANGE = re.compile(r"\b(?:between|from)\s+((?:19|20|21)\d\d)\s+(?:and|to|until|through)\s+((?:19|20|21)\d\d)\b")
RE_YEAR_CMP = re.compile(r"\b(since|after|before|until|through|prior to|by)\s+((?:19|20|21)\d\d)\b")
RE_YEAR = re.compile(r"\b((?:19|20|21)\d\d)\b")
RE_LATEST_YEAR = re.compile(r"\b(?:latest|newest|most recent|last)\s+year\b")

FIELD_WORDS = r"(orbits?|orbit types?|sizes?|size categor\w+|categor\w+|countr\w+|nations?|years?|risks?(?: levels?)?)"
RE_BREAKDOWN = re.compile(
    rf"\b(?:by|per|each|every|across|distribution|breakdown|split|spread)\b(?:\s+(?:of|the|debris|objects|in|over))*\s+{FIELD_WORDS}\b")
RE_COUNT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b")
RE_AVERAGE = re.compile(r"\b(average|mean|avg|median|typical)\b")
RE_EXTREME = re.compile(
    r"\b(largest|biggest|smallest|tiniest|highest|farthest|furthest|lowest|closest|"
    r"newest|latest|most recent|recent|oldest|earliest)\b")
RE_SIZE_WORDS = [("large", re.compile(r"\b(?:large|big|huge|massive)\b")),
                 ("medium", re.compile(r"\b(?:medium|mid ?sized?)\b")),
                 ("small", re.compile(r"\b(?:small|tiny|little)\b"))]
RE_LIMIT_A = re.compile(r"\b(?:top|first|show|list|display|find|give|get|only|limit)\s+(?:me\s+|us\s+)?(?:the\s+)?(\d{1,3})\b")
RE_LIMIT_B = re.compile(r"\b(\d{1,3})\s+(?:largest|biggest|smallest|newest|oldest|highest|lowest|latest|"
                        r"results?|objects?|items?|pieces?|records?|entries|rows|debris)\b")
RE_ALL = re.compile(r"\b(?:show|list|display|give|get)\s+(?:me\s+)?(?:all|every)\b|\beverything\b|\ball of them\b")
RE_NOUN = re.compile(r"\b(debris|objects?|junk|pieces?|items?|records?|entries|fragments?)\b")
RE_ID = re.compile(r"\bdeb\s*(\d{1,4})\b")
RE_HELP = re.compile(r"\bhelp\b|what can you do|\bexamples?\b|how do i (?:use|ask)|\bcommands\b|\bcapabilities\b")
RE_WHATIS = re.compile(r"\bwhat is (?:space )?(?:debris|junk)\b")
RE_THANKS = re.compile(r"^(?:thanks|thank you|thx|ty)\b")
GREETINGS = {"hi", "hello", "hey", "greetings", "yo", "salam", "assalam", "morning", "afternoon", "evening",
             "good", "there", "assistant", "bot", "ai", "everyone"}

STOPWORDS = set("""
show me us the a an of in on at for to from with and or that which are is were was be been there any all some
what whats how who where when why give list display find get see tell about do does did can could would please
i you we it its this these those debris object objects piece pieces junk item items record records space orbit
orbits orbiting altitude altitudes size sized sizes km category categories level levels type types added tracked
tracking catalogue catalog data dataset entries entry stats statistics statistic number numbers total count most
least only just also near around than then thanks thank top first last their them they has have had having into
out up down by per each across between since after before until through prior within under over above below
more less greater higher lower bigger larger smaller high low medium large small big tiny huge year years risk
risks risky country countries fragments fragment known tracked currently now currently currently found
""".split())

SUPERLATIVES = {
    "largest": ("size_cm", False, "largest"), "biggest": ("size_cm", False, "largest"),
    "smallest": ("size_cm", True, "smallest"), "tiniest": ("size_cm", True, "smallest"),
    "highest": ("altitude", False, "highest-altitude"), "farthest": ("altitude", False, "highest-altitude"),
    "furthest": ("altitude", False, "highest-altitude"),
    "lowest": ("altitude", True, "lowest-altitude"), "closest": ("altitude", True, "lowest-altitude"),
    "newest": ("year", False, "newest"), "latest": ("year", False, "newest"),
    "most recent": ("year", False, "newest"), "recent": ("year", False, "newest"),
    "oldest": ("year", True, "oldest"), "earliest": ("year", True, "oldest"),
}


def normalize(q: str) -> str:
    q = q.lower().strip().replace("’", "'")
    q = q.replace("u.s.a.", " usa ").replace("u.s.", " usa ")
    q = re.sub(r"\b(\d{4})\s*[-–—]\s*(\d{4})\b", r"between \1 and \2", q)
    q = re.sub(r"(?<=\d),(?=\d{3})", "", q)                    # 20,000 -> 20000
    q = q.replace(">=", " at least ").replace("<=", " at most ").replace(">", " over ").replace("<", " under ")
    q = re.sub(r"'s\b", "", q)                                 # what's -> what
    q = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", q)                  # keep decimal points only
    q = re.sub(r"[^a-z0-9. ]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _fmt_num(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}".rstrip("0")


def _unit_info(u: str) -> tuple[str, float, str]:
    """-> (data field, factor to field unit, display unit)."""
    if u.startswith("k"):
        return "altitude", 1.0, "km"
    if u.startswith("c"):
        return "size_cm", 1.0, "cm"
    if u == "mm" or u.startswith("milli"):
        return "size_cm", 0.1, "mm"
    return "size_cm", 100.0, "m"


def _join(items, word="or"):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


class DebrisAI:
    """Question -> exact answer."""

    def __init__(self, data: DebrisData):
        self.data = data

    # ------------------------------------------------------------------ public
    def ask(self, query: str) -> AIResponse:
        t0 = time.perf_counter()
        try:
            resp = self._ask(query)
        except Exception as exc:                      # never crash the UI
            resp = AIResponse(f"Sorry, I couldn't process that ({exc}). Try rephrasing.", kind="error")
        resp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return resp

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _help_text() -> str:
        return ("I answer questions about the debris catalogue using exact filters. You can combine:\n"
                "• Orbit: LEO, MEO, GEO, HEO  (or 'low earth', 'geostationary' ...)\n"
                "• Size: large / medium / small, or 'larger than 50 cm', 'under 2 m'\n"
                "• Altitude: 'above 1000 km', 'between 500 and 800 km'\n"
                "• Year: '2020', 'since 2019', 'before 2021', 'between 2019 and 2021'\n"
                "• Risk: high / medium / low   • Country: USA, Russia, China, ESA, India, Japan\n"
                "• Questions: 'how many ...', 'average altitude of ...', 'largest / newest / lowest ...'\n"
                "• Breakdowns: '... by orbit / size / country / year / risk'\n"
                "• Lookup: 'DEB-0042'   • Limits: 'top 5', 'show 20', 'show all'\n"
                "Try: " + " | ".join(f"“{e}”" for e in EXAMPLES[:3]))

    def _rows(self, idx) -> tuple[list, list]:
        d = self.data
        rows = [(str(d.name[i]), str(d.orbit[i]), format_size(d.size_cm[i]), f"{int(d.altitude[i]):,}",
                 str(d.country[i]), str(d.year[i]), str(d.risk[i]).title()) for i in idx]
        return rows, [int(d.id[i]) for i in idx]

    # -------------------------------------------------------------------- core
    def _ask(self, query: str) -> AIResponse:
        d = self.data
        text = normalize(query)
        if not text:
            return AIResponse("Ask me something about the debris, e.g. “" + EXAMPLES[1] + "”.", kind="help")
        if RE_THANKS.match(text):
            return AIResponse("You're welcome! Ask me anything else about the debris.", kind="greeting")
        if RE_HELP.search(text):
            return AIResponse(self._help_text(), kind="help")
        if RE_WHATIS.search(text):
            return AIResponse("Space debris is human-made junk in orbit — dead satellites, spent rocket stages "
                              "and fragments from collisions. Even a 1 cm fragment can destroy a satellite at orbital speed. "
                              "This catalogue holds " + f"{len(d):,} tracked objects. Ask me to filter or count them!", kind="help")

        # direct lookup by name: "DEB-0042"
        ids = [int(x) for x in RE_ID.findall(text)]
        if ids:
            idx = [int(np.flatnonzero(d.id == i)[0]) for i in ids if 1 <= i <= len(d)]
            if not idx:
                return AIResponse(f"There is no object with that ID (catalogue has DEB-0001 … DEB-{len(d):04d}).", kind="none")
            rows, rid = self._rows(idx)
            return AIResponse(f"Found {len(idx)} object{'s' if len(idx) > 1 else ''}.", kind="list",
                              columns=ROW_COLUMNS, rows=rows, record_ids=rid, total=len(idx))

        work = text
        m = d.all_mask()
        adjs: list[str] = []
        tails: list[str] = []
        interp: list[str] = []
        year_cov = f"{d.years[0]}–{d.years[-1]}"
        hints: list[str] = []

        # ---- numeric size / altitude comparisons (need units) -------------------------------
        def apply_cmp(fieldname, op, val, label):
            nonlocal m
            arr = getattr(d, fieldname)
            m &= {">": arr > val, ">=": arr >= val, "<": arr < val, "<=": arr <= val}[op]
            interp.append(label)

        def between_cb(mt):
            a, ua, b, ub = mt.groups()
            fname, fac, disp = _unit_info(ub)
            fa = _unit_info(ua)[1] if ua else fac
            lo, hi = float(a) * fa, float(b) * fac
            if lo > hi:
                lo, hi = hi, lo
            arr = getattr(d, fname)
            nonlocal m
            m &= (arr >= lo) & (arr <= hi)
            name = "altitude" if fname == "altitude" else "size"
            tails.append(f"with {name} {_fmt_num(float(a))}–{_fmt_num(float(b))} {disp}")
            interp.append(f"{name} {a}–{b} {disp}")
            return " "

        def cmp_cb(mt):
            opw, num, unit = mt.groups()
            op = next(o for o, pat in OP_WORDS.items() if re.fullmatch(pat, opw))
            fname, fac, disp = _unit_info(unit)
            val = float(num) * fac
            name = "altitude" if fname == "altitude" else "size"
            apply_cmp(fname, op, val, f"{name} {op} {_fmt_num(float(num))} {disp}")
            tails.append(f"with {name} {op} {_fmt_num(float(num))} {disp}")
            if fname == "altitude":
                hints.append("altitudes")
            return " "

        work = RE_BETWEEN.sub(between_cb, work)
        work = RE_CMP.sub(cmp_cb, work)

        # ---- year -----------------------------------------------------------------------------
        years_used = False

        def year_mask(sel_mask, label_tail, label_int):
            nonlocal m, years_used
            m &= sel_mask
            tails.append(label_tail)
            interp.append(label_int)
            years_used = True

        mt = RE_YEAR_RANGE.search(work)
        if mt:
            a, b = sorted((int(mt.group(1)), int(mt.group(2))))
            year_mask((d.year >= a) & (d.year <= b), f"added {a}–{b}", f"year {a}–{b}")
            work = work[:mt.start()] + " " + work[mt.end():]
        for mt in list(RE_YEAR_CMP.finditer(work)):
            word, y = mt.group(1), int(mt.group(2))
            if word == "since":
                year_mask(d.year >= y, f"added since {y}", f"year ≥ {y}")
            elif word == "after":
                year_mask(d.year > y, f"added after {y}", f"year > {y}")
            elif word in ("before", "prior to"):
                year_mask(d.year < y, f"added before {y}", f"year < {y}")
            else:                                           # until / through / by
                year_mask(d.year <= y, f"added up to {y}", f"year ≤ {y}")
        work = RE_YEAR_CMP.sub(" ", work)
        if RE_LATEST_YEAR.search(work):
            ly = d.years[-1]
            year_mask(d.year == ly, f"added in {ly}", f"year = {ly} (latest in data)")
            work = RE_LATEST_YEAR.sub(" ", work)
        bare = sorted({int(y) for y in RE_YEAR.findall(work)})
        if bare:
            year_mask(np.isin(d.year, bare), "added in " + _join(map(str, bare), "or"),
                      "year = " + _join(map(str, bare), "or"))
            work = RE_YEAR.sub(" ", work)

        # ---- categorical: orbit / risk / country -----------------------------------------
        def collect(aliases):
            nonlocal work
            found = []
            for code, pats in aliases:
                for p in pats:
                    work, n = re.subn(p, " ", work)
                    if n and code not in found:
                        found.append(code)
            return found

        orbits = collect(ORBIT_ALIASES)
        risks = collect(RISK_ALIASES)
        countries = collect(COUNTRY_ALIASES)
        if orbits:
            m &= np.isin(d.orbit, orbits)
            tails.append("in " + _join(orbits))
            interp.append("orbit = " + _join(orbits))
        if risks:
            m &= np.isin(d.risk, risks)
            adjs.append("/".join(risks) + "-risk")
            interp.append("risk = " + _join(risks))
        if countries:
            m &= np.isin(d.country, countries)
            tails.append("from " + _join(countries))
            interp.append("country = " + _join(countries))

        # ---- intent words ------------------------------------------------------------------
        breakdown_field = None
        mt = RE_BREAKDOWN.search(work)
        if mt:
            w = mt.group(1)
            breakdown_field = ("orbit" if w.startswith("orbit") else "country" if w.startswith(("countr", "nation"))
                               else "year" if w.startswith("year") else "risk" if w.startswith("risk")
                               else "size_cat")
            work = work[:mt.start()] + " " + work[mt.end():]

        want_count = bool(RE_COUNT.search(work))
        work = RE_COUNT.sub(" ", work)

        avg_kind = None
        mt = RE_AVERAGE.search(work)
        if mt and re.search(r"\b(?:altitude|height|size|sizes)\b", work):
            avg_kind = "median" if mt.group(1) == "median" else "mean"
            work = RE_AVERAGE.sub(" ", work)
        avg_field = "altitude" if re.search(r"\b(?:altitude|height)\b", work) else "size_cm"

        sup = None
        mt = RE_EXTREME.search(work)
        if mt:
            sup = SUPERLATIVES[mt.group(1)]
            work = RE_EXTREME.sub(" ", work)

        # ---- size words (after 'largest' etc. were consumed) ----------------------------------
        sizes = []
        for cat, rx in RE_SIZE_WORDS:
            if rx.search(work):
                sizes.append(cat)
                work = rx.sub(" ", work)
        if sizes:
            m &= np.isin(d.size_cat, sizes)
            adjs.insert(0, "/".join(sizes))
            interp.append("size = " + _join(sizes))

        # ---- limit ---------------------------------------------------------------------------
        limit = None
        mt = RE_LIMIT_A.search(work) or RE_LIMIT_B.search(work)
        if mt:
            limit = max(1, int(mt.group(1)))
            work = work[:mt.start()] + " " + work[mt.end():]
        if RE_ALL.search(work):
            limit = 1000
            work = RE_ALL.sub(" ", work)
        has_noun = bool(RE_NOUN.search(work))
        work = RE_NOUN.sub(" ", work)

        # ---- greeting / nothing understood -------------------------------------------------------
        recognised = bool(interp or breakdown_field or want_count or avg_kind or sup or has_noun)
        words = work.split()
        if not recognised:
            if words and words[0] in GREETINGS and all(w in GREETINGS for w in words):
                return AIResponse("Hello! I'm the Space Debris Assistant. Ask me to filter, count, rank or "
                                  "break down the catalogue — type “help” to see everything I understand.",
                                  kind="greeting")
            return AIResponse("I couldn't find anything to search for in that. " + self._help_text(), kind="none")

        ignored = [w for w in words if w not in STOPWORDS and w not in GREETINGS and len(w) > 2 and not w.isdigit()]
        ignored_note = f"\n(I ignored: {', '.join(ignored[:5])})" if ignored else ""

        # ---- selection ----------------------------------------------------------------------------
        idx = np.flatnonzero(m)
        total = int(idx.size)
        noun = " ".join(adjs + ["debris objects"] + tails)
        noun1 = noun.replace("debris objects", "debris object", 1)
        interp_text = ("Understood: " + " · ".join(interp)) if interp else "Understood: all objects"

        def zero_msg():
            tips = []
            if years_used:
                tips.append(f"the catalogue only covers {year_cov}")
            if "altitudes" in hints:
                tips.append("typical altitudes: LEO 200–1,200 km, MEO 2,000–35,000 km, GEO ≈35,786 km")
            return f"No {noun} found." + (" Note: " + "; ".join(tips) + "." if tips else "")

        def finish(resp):
            resp.interpretation = interp_text + (f" · limit = {limit}" if limit else "")
            resp.total = total
            resp.message += ignored_note
            return resp

        # ---- breakdown table ----------------------------------------------------------------------
        if breakdown_field:
            counts = d.count_by(breakdown_field, m)
            items = list(counts.items())
            if breakdown_field in ("orbit", "country"):
                items.sort(key=lambda kv: (-kv[1], kv[0]))
            label = {"orbit": "Orbit", "size_cat": "Size category", "country": "Country",
                     "year": "Year", "risk": "Risk"}[breakdown_field]
            rows = [(str(k).title() if breakdown_field in ("size_cat", "risk") else str(k), f"{v:,}",
                     f"{(v / total * 100 if total else 0):.1f}%") for k, v in items]
            if total == 0:
                return finish(AIResponse(zero_msg(), kind="none"))
            top = max(items, key=lambda kv: kv[1])
            msg = (f"Breakdown of {total:,} {noun} by {label.lower()}: "
                   f"{str(top[0]).title() if breakdown_field in ('size_cat', 'risk') else top[0]} "
                   f"is the largest group with {top[1]:,} ({top[1] / total * 100:.1f}%).")
            return finish(AIResponse(msg, kind="breakdown", columns=(label, "Objects", "Share"), rows=rows))

        # ---- how many -----------------------------------------------------------------------------
        if want_count:
            pct = total / len(d) * 100
            verb = "is" if total == 1 else "are"
            msg = (f"There {verb} {total:,} {noun if total != 1 else noun1}"
                   f" — {pct:.1f}% of the {len(d):,} objects in the catalogue.")
            if total == 0:
                msg = zero_msg()
            return finish(AIResponse(msg, kind="count"))

        # ---- average / median -----------------------------------------------------------------------
        if avg_kind:
            if total == 0:
                return finish(AIResponse(zero_msg(), kind="none"))
            arr = getattr(d, avg_field)[idx].astype(float)
            val = float(np.mean(arr) if avg_kind == "mean" else np.median(arr))
            if avg_field == "altitude":
                fmt = lambda v: f"{v:,.0f} km"
                name = "altitude"
            else:
                fmt = format_size
                name = "size"
            msg = (f"The {'average' if avg_kind == 'mean' else 'median'} {name} of {total:,} {noun} is "
                   f"{fmt(val)} (range {fmt(float(arr.min()))} to {fmt(float(arr.max()))}).")
            return finish(AIResponse(msg, kind="average"))

        # ---- ranked / plain list ---------------------------------------------------------------------
        if total == 0:
            return finish(AIResponse(zero_msg(), kind="none"))
        if sup:
            key, ascending, label = sup
            keyvals = getattr(d, key)[idx].astype(float)
            order = np.lexsort((d.id[idx], keyvals if ascending else -keyvals))
            idx = idx[order]
            limit = limit or 5
        else:
            limit = limit or 10
        shown = idx[:limit]
        rows, rid = self._rows(shown)

        if sup:
            top = d.record(int(shown[0]))
            if key == "year":
                ties = int(np.count_nonzero(d.year[idx] == top["year"]))
                msg = (f"The {label} objects in {noun} were added in {top['year']} ({ties:,} of {total:,}). "
                       f"First by ID: {top['name']} ({top['orbit']}, {format_size(top['size_cm'])}, {top['risk']} risk).")
            else:
                msg = (f"The {label} object among {total:,} {noun} is {top['name']}: "
                       f"{format_size(top['size_cm'])}, {top['orbit']} at {top['altitude']:,} km, "
                       f"added {top['year']}, {top['risk']} risk, from {top['country']}.")
            if len(shown) > 1:
                msg += f" Top {len(shown)} shown below."
        else:
            msg = f"Found {total:,} {noun if total != 1 else noun1}."
            if total > len(shown):
                msg += f" Showing the first {len(shown):,} (say “show all” or “top 50” for more)."
        return finish(AIResponse(msg, kind="list", columns=ROW_COLUMNS, rows=rows, record_ids=rid))
