"""Build data.js for the NHTS travel-behaviour report page (index.html).

Inputs
  final_data.csv  - cleaned NHTS 2017 trips (this folder, git-ignored)
  ../../Data/NHTS_transport_data/trippub.csv - raw NHTS 2017 trip file (outside the repo)

Why the raw file is needed: final_data's Transport_mode / WHYTO labels were taken from the
NHTS 2022 codebook, but the data is NHTS 2017, whose codes differ ("car" is really walking).
Joining back on household, person, minutes, miles and trip order recovers the true mode code
(TRPTRANS), the summary purpose (WHYTRP1S) and the trip weight (WTTRDFIN). All 817,981 rows match.

Every share on the page is weighted with WTTRDFIN (US trips per year), like official NHTS figures.

Run from portfolio-site/:  penv\\Scripts\\python mode_choice_analyics\\build_report_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "Data" / "NHTS_transport_data" / "trippub.csv"

f = pd.read_csv(HERE / "final_data.csv")
t = pd.read_csv(RAW, usecols=["HOUSEID", "PERSONID", "TRVLCMIN", "TRPMILES", "TRPTRANS", "WHYTRP1S", "WTTRDFIN"])
key = ["HOUSEID", "PERSONID", "TRVLCMIN", "TRPMILES"]
f["_n"] = f.groupby(key).cumcount()
t["_n"] = t.groupby(key).cumcount()
m = f.merge(t, on=key + ["_n"], how="left", validate="one_to_one")
assert m.TRPTRANS.notna().all(), "some final_data rows did not match trippub.csv"

MODE = {1: "Walk", 2: "Bicycle", 3: "Car", 4: "SUV", 5: "Van", 6: "Pickup truck", 7: "Golf cart / Segway",
        8: "Motorcycle", 9: "RV", 10: "School bus", 11: "Public bus", 12: "Paratransit", 13: "Charter / shuttle bus",
        14: "City-to-city bus", 15: "Amtrak / commuter rail", 16: "Subway / light rail", 17: "Taxi / ride-hail",
        18: "Rental car", 19: "Airplane", 20: "Boat / ferry", 97: "Something else"}
def group(c):
    if c in (3, 4, 5, 6): return "Private vehicle"
    if c == 1: return "Walk"
    if c in (11, 12, 14, 15, 16): return "Transit"
    if c == 2: return "Bicycle"
    return "Other"
GROUPS = ["Private vehicle", "Walk", "Transit", "Bicycle", "Other"]
PURPOSE = {1: "Home", 10: "Work", 20: "School / daycare / religious", 30: "Medical / dental", 40: "Shopping / errands",
           50: "Social / recreational", 70: "Transport someone", 80: "Meals", 97: "Something else"}

m["code"] = m.TRPTRANS.astype(int)
m["mode"] = m.code.map(MODE)
m["grp"] = m.code.map(group)
m["w"] = m.WTTRDFIN
W = m.w.sum()
pv = m.grp == "Private vehicle"
z = m.HHVEHCNT == 0

def share(mask, base=None):
    base = np.ones(len(m), bool) if base is None else base
    return float(m.w[mask & base].sum() / m.w[base].sum())

def mix(by, order):
    """Weighted mode-group mix (rows sum to 1) for each value of `by`."""
    out = []
    for v in order:
        b = by == v
        out.append({"label": str(v), "n": int(b.sum()), **{g: share(m.grp == g, b) for g in GROUPS}})
    return out

# distance bands
bins = [0, 1, 2, 5, 10, 25, np.inf]
blab = ["Under 1 mi", "1–2 mi", "2–5 mi", "5–10 mi", "10–25 mi", "25+ mi"]
m["dist"] = pd.cut(m.TRPMILES, bins, right=False, labels=blab)

# age bands
abins = [16, 25, 35, 45, 55, 65, 75, 200]
alab = ["16–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75+"]
m["age"] = pd.cut(m.R_AGE, abins, right=False, labels=alab)

m["veh"] = m.HHVEHCNT.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
m["purp"] = m.WHYTRP1S.map(PURPOSE)
purp_order = [PURPOSE[k] for k in (10, 20, 30, 40, 80, 50, 70, 1)]

lab = m.Transport_mode
data = {
    "trips": int(len(m)), "households": int(m.HOUSEID.nunique()),
    "persons": int(len(m.drop_duplicates(["HOUSEID", "PERSONID"]))),
    "weighted_trips_bn": float(W / 1e9),
    "kpi": {
        "private_vehicle": share(pv),
        "walk": share(m.grp == "Walk"),
        "transit": share(m.grp == "Transit"),
        "pv_under2": share(m.TRPMILES < 2, pv),
        "pv_under1": share(m.TRPMILES < 1, pv),
        "transit_zero_veh": share(m.grp == "Transit", z),
        "transit_with_veh": share(m.grp == "Transit", ~z),
        "walk_zero_veh": share(m.grp == "Walk", z),
        "zero_veh_trip_share": share(z),
    },
    "modes": sorted(({"mode": md, "grp": group(c), "share": share(m.code == c), "n": int((m.code == c).sum())}
                     for c, md in MODE.items()), key=lambda d: -d["share"]),
    "by_dist": mix(m.dist, blab),
    "by_veh": mix(m.veh, ["0", "1", "2", "3+"]),
    "by_age": mix(m.age, alab),
    "by_area": mix(m.URBRUR, ["Urban", "Rural"]),
    "by_purpose": mix(m.purp, purp_order),
    "pv_dist": [{"label": b, "share": share(m.dist == b, pv)} for b in blab],
    "claims": [
        {"claim": "Private vehicles carry 96% of trips",
         "labelled": float(lab.isin(["car", "suv", "pickup-truck"]).mean()), "corrected": share(pv), "fmt": "pct",
         "verdict": "Wrong", "why": "Walk trips were labelled \"car\"."},
        {"claim": "SUVs and pickups carry 87% of trips",
         "labelled": float(lab.isin(["suv", "pickup-truck"]).mean()), "corrected": share(m.code.isin([4, 6])), "fmt": "pct",
         "verdict": "Wrong", "why": "\"suv\" was really Car; \"pickup-truck\" was SUV + van + pickup."},
        {"claim": "31% of car trips are under 2 miles",
         "labelled": float(((m.TRPMILES < 2) & (lab == "suv")).sum() / (lab == "suv").sum()), "corrected": share(m.TRPMILES < 2, pv), "fmt": "pct",
         "verdict": "Close", "why": "About 29%, not 31%."},
        {"claim": "Zero-vehicle households take 5x more transit",
         "labelled": float((lab[z] == "other_mode").mean() / (lab[~z] == "other_mode").mean()),
         "corrected": share(m.grp == "Transit", z) / share(m.grp == "Transit", ~z), "fmt": "x",
         "verdict": "Understated", "why": "No transit label existed; with real codes it is about 15x."},
    ],
}
js = "window.MODE_DATA=" + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n"
(HERE / "data.js").write_text(js, encoding="utf-8")
print(f"wrote data.js ({len(js)/1024:.1f} KB)")
print(json.dumps({k: round(v, 4) for k, v in data["kpi"].items()}, indent=1))
for c in data["claims"]: print(c["claim"], round(c["labelled"], 4), round(c["corrected"], 4))
print("dist", [(d["label"], round(d["Private vehicle"], 3), round(d["Walk"], 3)) for d in data["by_dist"]])
print("veh", [(d["label"], round(d["Transit"], 3), round(d["Walk"], 3)) for d in data["by_veh"]])
print("age", [(d["label"], round(d["Private vehicle"], 3)) for d in data["by_age"]])
print("area", [(d["label"], round(d["Private vehicle"], 3), round(d["Transit"], 3)) for d in data["by_area"]])
print("purp", [(d["label"], round(d["Private vehicle"], 3), round(d["Walk"], 3)) for d in data["by_purpose"]])
print("modes", [(d["mode"], round(d["share"], 4)) for d in data["modes"][:9]])
