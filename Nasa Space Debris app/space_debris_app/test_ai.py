"""Checks the AI agent's answers against independent brute-force Python loops.

Run:  python test_ai.py
"""
import time

from debris_ai import DebrisAI
from debris_data import DebrisData

d = DebrisData()
ai = DebrisAI(d)
recs = [d.record(i) for i in range(len(d))]          # plain dicts: independent of the NumPy filters
fails = 0


def check(name, got, expected):
    global fails
    ok = got == expected
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got={got!r} expected={expected!r}"))


def total(q):
    return ai.ask(q).total


# --- counts / filters -------------------------------------------------------------------------
check("LEO count", total("how many debris in LEO"), sum(r["orbit"] == "LEO" for r in recs))
check("near earth = LEO", total("how many near earth"), sum(r["orbit"] == "LEO" for r in recs))
check("LEO + 2020", total("Show me debris near Earth orbit from 2020 show all"),
      sum(r["orbit"] == "LEO" and r["year"] == 2020 for r in recs))
check("high risk GEO", total("how many high-risk objects are in GEO?"),
      sum(r["orbit"] == "GEO" and r["risk"] == "high" for r in recs))
check("medium earth orbit is NOT medium size", total("how many in medium earth orbit"),
      sum(r["orbit"] == "MEO" for r in recs))
check("medium size", total("how many medium debris"), sum(r["size_cat"] == "medium" for r in recs))
check("large only (not 'largest')", total("how many large debris"), sum(r["size_cat"] == "large" for r in recs))
check("Russian", total("how many russian debris"), sum(r["country"] == "Russia" for r in recs))
check("USA + China", total("how many debris from usa or china"),
      sum(r["country"] in ("USA", "China") for r in recs))
check("size > 50 cm", total("how many larger than 50 cm"), sum(r["size_cm"] > 50 for r in recs))
check("size < 2 m", total("how many smaller than 2 m"), sum(r["size_cm"] < 200 for r in recs))
check("altitude > 1000 km", total("how many above 1000 km"), sum(r["altitude"] > 1000 for r in recs))
check("altitude between", total("how many between 500 and 800 km"), sum(500 <= r["altitude"] <= 800 for r in recs))
check("20,000 km thousands separator", total("how many over 20,000 km"), sum(r["altitude"] > 20000 for r in recs))
check("since 2020", total("how many since 2020"), sum(r["year"] >= 2020 for r in recs))
check("before 2020", total("how many before 2020"), sum(r["year"] < 2020 for r in recs))
check("range 2019-2021", total("how many 2019-2021"), sum(2019 <= r["year"] <= 2021 for r in recs))
check("between years", total("how many between 2019 and 2021"), sum(2019 <= r["year"] <= 2021 for r in recs))
check("year out of range -> 0", total("how many from 2035"), 0)
check("combo", total("how many russian debris larger than 50 cm since 2019 in LEO"),
      sum(r["country"] == "Russia" and r["size_cm"] > 50 and r["year"] >= 2019 and r["orbit"] == "LEO" for r in recs))
check("greeting 'this' does not trigger", ai.ask("which debris in this orbit LEO").kind, "list")

# --- rankings ---------------------------------------------------------------------------------
resp = ai.ask("What's the largest piece of debris in LEO?")
best = max((r for r in recs if r["orbit"] == "LEO"), key=lambda r: (r["size_cm"], -r["id"]))
check("largest in LEO is correct", resp.rows[0][0], best["name"])
resp = ai.ask("smallest debris in GEO")
best = min((r for r in recs if r["orbit"] == "GEO"), key=lambda r: (r["size_cm"], r["id"]))
check("smallest in GEO", resp.rows[0][0], best["name"])
resp = ai.ask("highest altitude debris")
best = max(recs, key=lambda r: (r["altitude"], -r["id"]))
check("highest altitude", resp.rows[0][0], best["name"])
resp = ai.ask("top 3 largest")
exp = sorted(recs, key=lambda r: (-r["size_cm"], r["id"]))[:3]
check("top 3 largest", [r[0] for r in resp.rows], [r["name"] for r in exp])
check("show 2020 is a year, not a limit", len(ai.ask("show debris from 2020").rows), 10)
check("show all", len(ai.ask("show all debris").rows), 1000)

# --- aggregates ----------------------------------------------------------------------------------
resp = ai.ask("average altitude of LEO debris")
vals = [r["altitude"] for r in recs if r["orbit"] == "LEO"]
check("average altitude text", f"{sum(vals) / len(vals):,.0f} km" in resp.message, True)
resp = ai.ask("Show debris by size category")
exp = [(c, sum(r["size_cat"] == c for r in recs)) for c in ("small", "medium", "large")]
check("breakdown by size", [(r[0].lower(), int(r[1].replace(",", ""))) for r in resp.rows], exp)
resp = ai.ask("how many per orbit in 2021")
exp = {o: sum(r["orbit"] == o and r["year"] == 2021 for r in recs) for o in ("LEO", "MEO", "GEO", "HEO")}
check("breakdown by orbit + year", {r[0]: int(r[1].replace(",", "")) for r in resp.rows}, exp)

# --- misc -----------------------------------------------------------------------------------------
check("lookup", ai.ask("DEB-0042").rows[0][0], "DEB-0042")
check("hello", ai.ask("hello").kind, "greeting")
check("help", ai.ask("help").kind, "help")
check("gibberish", ai.ask("blah blah").kind, "none")
check("unknown country is flagged", "ignored" in ai.ask("show debris from brazil").message, True)

# --- speed ----------------------------------------------------------------------------------------
t = time.perf_counter()
for _ in range(200):
    ai.ask("Russian debris larger than 50 cm in LEO since 2019, top 5 largest")
per = (time.perf_counter() - t) / 200 * 1000
print(f"\nSpeed: {per:.3f} ms per query")
print("ALL TESTS PASSED" if not fails else f"{fails} TEST(S) FAILED")
raise SystemExit(1 if fails else 0)
