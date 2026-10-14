"""Read-only probes for full-sweep-16 A1 (equations/math + net tools).

Run from OUTSIDE the repo dir:
    .venv/Scripts/python.exe -I docs/exploration/full-sweep-16/_probe_a1.py

No network. No writes. Payloads are read-only (os.getcwd / os.listdir of a
nonexistent path).
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import os
import sys
import traceback
import types
from contextlib import redirect_stdout

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
TOOLS = os.path.join(REPO, "src", "backend", "tools")
sys.path.insert(0, REPO)

# Synthetic package so intra-directory relative imports (from ._http import ...) work.
_pkg = types.ModuleType("ptools")
_pkg.__path__ = [TOOLS]
sys.modules["ptools"] = _pkg


def load(name: str):
    return importlib.import_module(f"ptools.{name}")


def head(title: str) -> None:
    print(f"\n===== {title} =====")


# ---------------------------------------------------------------- sympy facts
head("P1 sympy.factorint / primefactors sign & zero semantics")
import sympy  # noqa: E402

for n in (-6, 6, 0, 1, -1, -12, 12):
    try:
        print(f"factorint({n}) = {sympy.factorint(n)}")
    except Exception as exc:  # noqa: BLE001
        print(f"factorint({n}) RAISED {type(exc).__name__}: {exc}")
for n in (0, 1, -1, -6, 6):
    try:
        print(f"primefactors({n}) = {sympy.primefactors(n)}")
    except Exception as exc:  # noqa: BLE001
        print(f"primefactors({n}) RAISED {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- number_theory
nt = load("number_theory_tool")
head("P2 number_theory_tool: sign / zero / uncaught paths")
for op, kw in [
    ("factorize", {"number": -6}),
    ("factorize", {"number": 0}),
    ("prime_factors", {"number": 0}),
    ("prime_factors", {"number": -6}),
    ("is_prime", {"number": -7}),
    ("next_prime", {"number": 0}),
    ("gcd", {"number": 0, "second_number": 0}),
    ("lcm", {"number": 0, "second_number": 0}),
    ("to_base", {"number": -255, "base": 16}),
    ("to_base", {"number": 0, "base": 2}),
]:
    try:
        print(f"{op:14s} {kw} -> {nt.number_theory(operation=op, **kw)!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{op:14s} {kw} UNCAUGHT {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- math_tool RCE
mt = load("math_tool")
MARKER = os.path.join(os.environ.get("TEMP", "/tmp"), "sweep16_readonly_marker.txt")
try:
    os.remove(MARKER)
except OSError:
    pass
PAYLOAD = (
    "__import__('os').getcwd()"  # read-only: returns the CWD, writes nothing
)

head("P3 math_tool RCE reachability (read-only payload)")
print("payload:", PAYLOAD)
cases = [
    dict(operation="evaluate", expression=PAYLOAD),
    dict(operation="solve", expression=f"{PAYLOAD} = 0"),
    dict(operation="derivative", expression=PAYLOAD),
    dict(operation="simplify", expression=PAYLOAD),
    dict(operation="limit", expression="1/x", point=PAYLOAD),
    dict(operation="series", expression="exp(x)", point=PAYLOAD),
    dict(operation="integral", expression="x", lower_bound=PAYLOAD, upper_bound=PAYLOAD),
]
for kw in cases:
    try:
        out = mt.solve_math(**kw)
        marker_exists = os.path.exists(MARKER)
        print(f"{kw.get('operation'):11s} -> marker={marker_exists} out={out[:150]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{kw.get('operation'):11s} UNCAUGHT {type(exc).__name__}: {exc}")

head("P3b parse_expr default namespace exposes builtins?")
from sympy.parsing.sympy_parser import parse_expr  # noqa: E402

v = parse_expr("__import__")
print("__import__ ->", type(v).__module__, type(v).__name__, str(v)[:60])
for expr in ["os.system", "getattr(1,'a')"]:
    try:
        print(f"{expr:14s} -> {str(parse_expr(expr))[:70]}")
    except Exception as exc:  # noqa: BLE001
        print(f"{expr:14s} RAISED {type(exc).__name__}")

head("P3c math_tool: int-returning read-only payload = SILENT success?")
for expr in [
    "__import__('os').getpid()",
    "__import__('os').getppid()",
    "__import__('sys').maxsize",
    "__import__('len').__self__",
    "min(1,2)",
]:
    try:
        out = mt.solve_math(operation="evaluate", expression=expr)
        print(f"   {expr:28s} -> {out[:120]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {expr:28s} UNCAUGHT {type(exc).__name__}: {exc}")
print("   real getpid for reference:", os.getpid())

head("P3d sympy constructors that re-sympify their string arg")
ns = {}
exec("from sympy import *", ns)
ns["__builtins__"] = {}
for fn in ["nsimplify", "solveset", "Poly", "Function", "Piecewise", "Rational", "Integer"]:
    try:
        r = parse_expr(f"{fn}('1/2')", global_dict=ns, evaluate=True)
        print(f"   ns+{fn:11s}('1/2') -> {type(r).__name__}: {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ns+{fn:11s}('1/2') RAISED {type(exc).__name__}")

head("P3e IS THE RECOMMENDED FIX STILL BYPASSABLE?")
print("sandbox = exec('from sympy import *', ns); ns['__builtins__']={}; global_dict=ns")
print("real parent pid = ", os.getppid())
P = "__import__(" + chr(39) + "os" + chr(39) + ").getppid()"
DQ = chr(34)
SQ = chr(39)
for expr in [
    P,
    "nsimplify(" + DQ + P + DQ + ")",
    "compile(" + DQ + P + DQ + ",0," + SQ + "eval" + SQ + ")",
    "nsimplify(" + SQ + "1/2" + SQ + ")",
    "sin(0) + 1",
]:
    try:
        r = parse_expr(expr, global_dict=ns, evaluate=True)
        print(f"   {expr[:46]:46s} -> {type(r).__name__} = {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {expr[:46]:46s} BLOCKED {type(exc).__name__}")

# ---------------------------------------------------------------- linalg RCE
la = load("linalg_tool")
head("P4 linalg_tool RCE reachability (read-only payload)")
CELL = '"__import__(\'os\').getppid()"'  # ast.literal_eval-safe string cell
for op, kw in [
    ("rank", {"matrix": f"[[1, 2], [3, {CELL}]]"}),
    ("solve", {"matrix": "[[1, 2], [3, 4]]", "vector": f"[5, {CELL} ]"}),
    ("multiply", {"matrix": "[[1, 2], [3, 4]]", "matrix_b": f"[[1, {CELL}],[3,4]]"}),
    ("determinant", {"matrix": f"[[1, {CELL}],[3,4]]"}),
    ("transpose", {"matrix": f"[[{CELL}, 2],[3,4]]"}),
    ("eigenvalues", {"matrix": f"[[{CELL}, 2],[3,4]]"}),
]:
    try:
        out = la.linear_algebra(operation=op, **kw)
        print(f"{op:12s} -> {out[:140]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{op:12s} UNCAUGHT {type(exc).__name__}: {exc}")

head("P4b linalg: dimension bound / ragged / bool / float")
for op, kw in [
    ("rank", {"matrix": "[[True]]"}),
    ("rank", {"matrix": "[[1.5]]"}),
    ("rank", {"matrix": "[]"}),
    ("rank", {"matrix": "[[1,2],[3]]"}),
    ("rank", {"matrix": "{'a': 1}"}),
    ("multiply", {"matrix": "[[1]]", "matrix_b": "[[1],[2]]"}),
]:
    try:
        print(f"{op:10s} {str(kw)[:52]:54s} -> {la.linear_algebra(operation=op, **kw)[:90]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{op:10s} UNCAUGHT {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- math_tool behaviour
head("P5 math_tool behavioural semantics")
beh = [
    dict(operation="integral", expression="x", lower_bound="0"),
    dict(operation="integral", expression="x", upper_bound="1"),
    dict(operation="limit", expression="1/x", point="0", direction="garbage"),
    dict(operation="limit", expression="1/x", point="0", direction="-  "),
    dict(operation="series", expression="exp(x)", order=20),
    dict(operation="solve", expression="a = b = c"),
    dict(operation="solve", expression="2*x + 1 = 5"),
    dict(operation="evaluate", expression="1/0"),
    dict(operation="evaluate", expression="2**99999"),
    dict(operation="derivative", expression="sin(x)", variable="sin(x)"),
    dict(operation="derivative", expression="f(x)", variable="f"),
    dict(operation="evaluate", expression="Integer(1) + 2"),
    dict(operation="evaluate", expression="nsimplify('0')"),
]
for name in ["doctest", "factorint", "solveset", "exec", "eval", "open", "compile"]:
    try:
        v = parse_expr(name)
        print(f"   parse_expr({name!r:11s}) -> {v.__class__.__name__} callable={callable(v)}")
    except Exception as exc:  # noqa: BLE001
        print(f"   parse_expr({name!r:11s}) RAISED {type(exc).__name__}")
for kw in beh:
    try:
        out = mt.solve_math(**kw)
        print(f"{kw.get('operation'):11s} {str({k:v for k,v in kw.items() if k!='operation'})[:44]:46s} -> {out[:150]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{kw.get('operation'):11s} UNCAUGHT {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- statistics
st = load("statistics_tool")
head("P6 statistics_tool: inf / nan / overflow / ReDoS")
for nums in ["1e999", "-1e999", "1e-999", "nan", "inf", "1, 2, abc, 4", ".", "e", "1,2,3,"]:
    try:
        print(f"{nums:12s} -> {st.compute_statistics(nums)[:110]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{nums:12s} UNCAUGHT {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- datetime
dt = load("datetime_tool")
head("P7 datetime_tool: difference / Z / awareness")
for kw in [
    dict(operation="difference", start="2026-06-30T15:00", end="2026-06-30T11:00"),
    dict(operation="difference", start="2026-06-30T15:00", end="2026-07-01T17:00"),
    dict(operation="difference", start="2026-06-30T15:00", end="2026-07-01T17:00+08:00"),
    dict(operation="difference", start="2026-06-30T15:00:00.999", end="2026-07-01T17:00"),
    dict(operation="convert_timezone", start="2026-06-30T15:00", timezone_name="UTC", to_timezone="Asia/Tokyo"),
    dict(operation="now"),
    dict(operation="add", start="2026-06-30T23:30", hours=2),
    dict(operation="convert_timezone", start="2026-03-08T02:30", timezone_name="America/New_York", to_timezone="UTC"),
]:
    try:
        print(f"{kw.get('operation'):17s} {str({k:v for k,v in kw.items() if k!='operation'})[:60]:62s} -> {dt.calculate_datetime(**kw)[:180]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{kw.get('operation'):17s} UNCAUGHT {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- _amap / _geocoding
amap = load("_amap")
geo = load("_geocoding")

head("P8 _amap: parse / mode / bounds / optional_float overflow")
print("parse_latlon('121.47,31.23') =", amap.parse_latlon("121.47,31.23"))
print("parse_latlon('31.23,121.47') =", amap.parse_latlon("31.23,121.47"))
print("parse_lnglat('1,2,3')        =", amap.parse_lnglat("1,2,3"))
print("parse_lnglat([100.0,20.0])   =", amap.parse_lnglat([100.0, 20.0]))
print("normalize_travel_mode checks:")
for m in ["bus", "driving", "garbage", "", None, "  WALK  ", "airplane", "bike"]:
    print(f"   {str(m):12s} -> {amap.normalize_travel_mode(m)!r}")
print("route_bounds cross antimeridian:",
      amap.route_bounds([amap.AMapPosition(179, 0), amap.AMapPosition(-179, 0)]))
print("_sample_positions(n=7,limit=5):",
      [p.lng for p in amap._sample_positions([amap.AMapPosition(float(i), 0) for i in range(7)], 5)])
print("_sample_positions(n=6,limit=5):",
      [p.lng for p in amap._sample_positions([amap.AMapPosition(float(i), 0) for i in range(6)], 5)])
print("_sample_positions(n=8,limit=4):",
      [p.lng for p in amap._sample_positions([amap.AMapPosition(float(i), 0) for i in range(8)], 4)])
print("parse_polyline max_points=0:", amap.parse_polyline("1,2;3,4", max_points=0))
for bad in ["1e999", "9" * 400, "9" * 10000, -5, "abc", None, [1, 2]]:
    try:
        print(f"_optional_float({str(bad)[:18]:20s}) -> {amap._optional_float(bad)!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"_optional_float({str(bad)[:18]:20s}) UNCAUGHT {type(exc).__name__}: {exc}")

head("P8b _amap: envelope validation leniency + secret redaction")
for payload, ctx in [
    ({"pois": []}, "pois-no-status"),
    ({"status": "0", "info": "key=SECRET123 leaked"}, "bad-status"),
    ({"errcode": "10001", "errmsg": "key=SECRET123 bad key"}, "v4-err"),
    ({"route": {"paths": []}}, "route-ok"),
]:
    try:
        print(f"{ctx:14s} -> {amap.validate_amap_envelope(payload, context=ctx)!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{ctx:14s} -> {type(exc).__name__}: {exc}")
print("safe_amap_error:", amap.safe_amap_error("boom ?key=Abc123&foo=bar"))
print("safe_amap_error:", amap.safe_amap_error("{'key': 'vK7secret', 'x': 1}"))
print("safe_amap_error:", amap.safe_amap_error("AMAP_WEB_SERVICE_KEY=vK7secret here"))

head("P8c _amap parse_route_result with hostile-but-plausible payload")
hostile = {
    "status": "1",
    "route": {"paths": [{
        "distance": "9" * 400,
        "duration": "1e999",
        "cost": {"duration": "9" * 400},
        "polyline": "",
        "steps": [{
            "instruction": "go key=hunterSecret now",
            "road": "r1",
            "distance": "100",
            "duration": "9" * 400,
            "polyline": "120.0,30.0;120.1,30.1;120.2,30.2",
        }],
    }]},
}
try:
    r = amap.parse_route_result(hostile, mode="driving")
    print("distance_m:", r.distance_m, "duration_s:", r.duration_s)
    print("steps:", len(r.steps), "step0:", r.steps[0].instruction if r.steps else None)
    print("bounds:", r.bounds)
except Exception as exc:  # noqa: BLE001
    print("UNCAUGHT", type(exc).__name__, exc)

head("P9 _geocoding: match score vs APPROXIMATE MATCH gate")
print("CONFIDENT_MATCH_SCORE =", geo.CONFIDENT_MATCH_SCORE)
for q, name, city in [
    ("上海", "上海浦东软件园祖冲之园", "上海"),
    ("上海", "随便一个楼", "上海"),
    ("apple", "Pineapple Farm", ""),
    ("a", "Apple Pie Shop At Home", ""),
    ("北京饭店", "北京饭店", "北京"),
]:
    s_name = geo.name_match_score(q, name)
    s_city = geo.name_match_score(q, city)
    print(f"q={q!r:12s} name={name!r:24s} city={city!r:6s} "
          f"-> name_score={s_name:.2f} city_score={s_city:.2f} max={max(s_name,s_city):.2f}")

head("P9b _geocoding: geocode_place with a fake requester (no network)")
fake_pois = [{"name": "随便一个楼", "location": "121.4,31.2", "cityname": "上海",
              "pname": "上海", "adname": "浦东", "address": "浦东新区某某路1号"}]
calls = []


def fake_get(url, **kwargs):
    calls.append((url, dict(kwargs)))
    class _R:
        def raise_for_status(self):  # noqa: D102
            return None
        def json(self):
            return {"status": "1", "pois": fake_pois}
    return _R()


places = geo.geocode_place("上海", limit=3, requester=fake_get, api_key="FAKE_KEY")
print("http calls made:", len(calls))
for p in places:
    print(f"  {p.label!r} score={p.match_score:.2f} is_confident={p.is_confident} source={p.source}")

head("P9c _geocoding: clean_place_query + dead-code check in _place_from_amap")
for q in ["在地图上找出上海的位置", "where is the apple pie shop", "地图", "the", "的"]:
    print(f"   {q!r:34s} -> {geo.clean_place_query(q)!r}")
try:
    r = amap.AMapLookupResult(name="", position=amap.AMapPosition(1, 2), source="poi")
    out = geo._place_from_amap(r, "")
    print("_place_from_amap(name='', term='') ->", None if out is None else out.label)
except Exception as exc:  # noqa: BLE001
    print("_place_from_amap UNCAUGHT", type(exc).__name__, exc)

# ---------------------------------------------------------------- currency / wikipedia / weather (offline)
cur = load("currency")
head("P10 currency: charset / rate paths (no network)")
for src, tgt in [("../", "EUR"), ("USD", "EUR"), ("USD", "../../../etc/passwd")]:
    print(f"   build path would be: {cur.CURRENCY_API_URL}/{src.strip().upper()}/{tgt.strip().upper()}")
for payload in [{"rate": "1.09", "date": "2026-10-09"}, {"rate": 0}, {"rate": "abc"}, {"rate": None}]:
    def _req(url, **kw):
        return payload
    try:
        print(f"   payload={str(payload)[:28]:30s} -> {cur.convert_currency(100.0, 'USD', 'EUR', requester=_req)[:100]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"   payload={str(payload)[:28]:30s} UNCAUGHT {type(exc).__name__}: {exc}")
print("   same-currency short circuit ->", cur.convert_currency(100.0, "usd", "usd"))

wk = load("wikipedia_tool")
head("P11 wikipedia: trim / url / hostile payload shapes")
for t in ["Title", "A B", "AT&T/LTE", "a/b/c?x=1", "日本語", ""]:
    print(f"   _page_url({t!r:14s}) -> {wk._page_url(t)!r}")
print("   _trim_summary long ->", repr(wk._trim_summary("x " * 1000, 20)))
print("   _trim_summary newline ->", repr(wk._trim_summary("a\nb\nc", 100)))

def _req_list(url, **kw):
    return {"query": {"search": [{"pageid": 1, "title": "T"}]}}
def _req_bad_search(url, **kw):
    return {"query": {"search": "not-a-list"}}
def _req_bad_pages(url, **kw):
    return {"query": {"pages": [{"title": "T"}]}}
def _req_good(url, **kw):
    calls2.append(url)
    if "srsearch" in dict(kw.get("params", {})):
        return {"query": {"search": [{"pageid": 7, "title": "Seven"}]}}
    return {"query": {"pages": {"7": {"title": "Seven", "extract": "Hello world"}}}}
for label, req in [("list", _req_list), ("search=str", _req_bad_search),
                   ("pages=list", _req_bad_pages), ("good", _req_good)]:
    calls2 = []
    try:
        print(f"   {label:12s} -> {wk.search_wikipedia('q', requester=req)[:110]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:12s} UNCAUGHT {type(exc).__name__}: {exc}")

wr = load("weather")
head("P12 weather: description / resolve / table")
for code in [0, 95, 96, 99, 95.0, 46, None, "bad"]:
    print(f"   _weather_description({code!r:6s}) -> {wr._weather_description(code)!r}")
for label, payload in [
    ("no-units", {"current": {"temperature_2m": 20.5}}),
    ("full", {"current": {"temperature_2m": 20.5, "relative_humidity_2m": 60,
                          "wind_speed_10m": 12, "weather_code": 95},
              "current_units": {"temperature_2m": "F", "relative_humidity_2m": "%", "wind_speed_10m": "km/h"},
              "daily": {"time": [f"d{i}" for i in range(16)],
                        "temperature_2m_max": [20] * 16, "temperature_2m_min": [10] * 16,
                        "weather_code": [95] * 16},
              "daily_units": {"temperature_2m_max": "C"}}),
    ("ragged-daily", {"daily": {"time": ["d0", "d1"], "temperature_2m_max": [20]}}),
]:
    try:
        out = wr._format_weather(label, payload)
        days = out.count("| d") if "d0" in out else 0
        print(f"   {label:13s} -> {len(out)} chars, table rows={days}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:13s} UNCAUGHT {type(exc).__name__}: {exc}")

head("P12b weather: resolve_location with hostile geojson")
def _geo_missing_keys(url, **kw):
    return {"results": [{"name": "X"}]}
try:
    print("   missing lat key ->", wr._resolve_location(city="X", latitude=None, longitude=None, requester=_geo_missing_keys))
except Exception as exc:  # noqa: BLE001
    print("   missing lat key UNCAUGHT", type(exc).__name__, exc)
print("   only latitude    ->", wr._resolve_location(city=None, latitude=31.2, longitude=None, requester=None))
print("   only longitude   ->", wr._resolve_location(city=None, latitude=None, longitude=121.4, requester=None))
print("   nothing          ->", wr._resolve_location(city=None, latitude=None, longitude=None, requester=None))

# ---------------------------------------------------------------- stock
sk = load("stock")
head("P13 stock: hostile ticker-object shapes (no network)")
class _T:
    def __init__(self, fi, info):
        self.fast_info = fi
        self.info = info
for label, fi, info in [
    ("normal", {"last_price": 100.0, "previous_close": 99.0, "currency": "USD"}, {}),
    ("strings", {"last_price": "1,234.56", "previous_close": "99.0", "currency": "USD"}, {}),
    ("none-price", {"last_price": None}, {}),
    ("volume-str", {"last_price": 1.0, "last_volume": "1,000", "currency": "USD"}, {}),
    ("nan-prev", {"last_price": 100.0, "previous_close": float("nan"), "currency": "USD"}, {}),
    ("zero-prev", {"last_price": 100.0, "previous_close": 0.0, "currency": "USD"}, {}),
    ("neg-prev", {"last_price": 100.0, "previous_close": -5.0, "currency": "USD"}, {}),
    ("inf-price", {"last_price": float("inf"), "currency": "USD"}, {}),
]:
    try:
        out = sk.get_stock_quote("AAPL", ticker_factory=lambda s: _T(fi, info))
        print(f"   {label:10s} -> {out[:150]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:10s} UNCAUGHT {type(exc).__name__}: {exc}")

print("\nDONE")
