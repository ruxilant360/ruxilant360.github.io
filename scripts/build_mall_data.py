"""Build mall-report/data.json from the raw CSVs.

Runs the month-end data checks, applies the corrections, and writes both the
as-reported and corrected figures so the report page can switch between them.

Usage: python scripts/build_mall_data.py   (from the portfolio-site folder)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "mall-report" / "data"
OUT = ROOT / "mall-report" / "data.json"

AS_OF = pd.Timestamp("2026-09-30")  # last closed month

props = pd.read_csv(DATA / "properties.csv")
units = pd.read_csv(DATA / "units.csv")
tenants = pd.read_csv(DATA / "tenants.csv")
leases = pd.read_csv(DATA / "leases.csv", parse_dates=["lease_start", "open_date", "lease_end"])
sales = pd.read_csv(DATA / "tenant_sales.csv", parse_dates=["sales_month"])
budget = pd.read_csv(DATA / "budget.csv", parse_dates=["sales_month"])
traffic = pd.read_csv(DATA / "traffic_daily.csv", parse_dates=["visit_date"])

months = pd.date_range("2025-01-01", AS_OF, freq="MS")
month_idx = {m: i for i, m in enumerate(months)}
key = ["unit_id", "tenant_id", "sales_month"]
issues = []

# ---------------------------------------------------------------- tenant sales checks
sales["flag"] = ""

# Check 1: duplicate submissions (same unit, tenant and month more than once)
dup_mask = sales.duplicated(key, keep="first")
for _, r in sales[dup_mask].iterrows():
    issues.append({
        "check": "Duplicate submission",
        "property_id": r.property_id,
        "where": f"{r.unit_id} · {r.sales_month:%b %Y}",
        "detail": f"Same tenant and month reported twice at ${r.reported_sales:,.0f} each.",
        "impact": -round(r.reported_sales, 2),
        "fix": "Removed the second row.",
    })
sales.loc[sales.duplicated(key, keep=False), "flag"] = "duplicate"
fixed = sales[~dup_mask].copy()
fixed["fixed_sales"] = fixed.reported_sales

# Check 2: month-over-month spikes against the tenant's own trailing 3-month average
fixed = fixed.sort_values(key)
grp = fixed.groupby(["unit_id", "tenant_id"]).reported_sales
fixed["trail3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
fixed["ratio"] = fixed.reported_sales / fixed.trail3
for idx, r in fixed[fixed.ratio > 5].iterrows():
    corrected = round(r.reported_sales / 10, 2)
    fixed.loc[idx, "fixed_sales"] = corrected
    fixed.loc[idx, "flag"] = "decimal slip"
    issues.append({
        "check": "Spike vs tenant's own history",
        "property_id": r.property_id,
        "where": f"{r.unit_id} · {r.sales_month:%b %Y}",
        "detail": f"Reported ${r.reported_sales:,.0f}, {r.ratio:.1f}× its 3-month average of ${r.trail3:,.0f}. "
                  f"Looks like a misplaced decimal (×10).",
        "impact": round(corrected - r.reported_sales, 2),
        "fix": f"Divided by 10 to ${corrected:,.0f}, pending tenant confirmation.",
    })

# Check 3: open leases with no report for the last closed month
last_m = months[-1]
m_end = last_m + pd.offsets.MonthEnd(0)
open_leases = leases[(leases.open_date <= m_end) & (leases.lease_end >= last_m)]
got = fixed[fixed.sales_month == last_m][["unit_id", "tenant_id"]]
missing = open_leases.merge(got, how="left", indicator=True).query("_merge == 'left_only'")
prev_m = months[-2]
for _, r in missing.iterrows():
    u = units.set_index("unit_id").loc[r.unit_id]
    prev = fixed[(fixed.unit_id == r.unit_id) & (fixed.sales_month == prev_m)].fixed_sales.sum()
    # Estimate: last month's sales × how much reporting tenants in the same mall changed month over month
    same = fixed[(fixed.property_id == u.property_id)]
    a = same[same.sales_month == last_m].set_index(["unit_id", "tenant_id"]).fixed_sales
    b = same[same.sales_month == prev_m].set_index(["unit_id", "tenant_id"]).fixed_sales
    both = a.index.intersection(b.index)
    mom = a.loc[both].sum() / b.loc[both].sum()
    est = round(prev * mom, 2)
    fixed = pd.concat([fixed, pd.DataFrame([{
        "property_id": u.property_id, "unit_id": r.unit_id, "tenant_id": r.tenant_id,
        "sales_month": last_m, "reported_sales": float("nan"), "fixed_sales": est, "flag": "missing",
    }])], ignore_index=True)
    issues.append({
        "check": "Open lease, no sales report",
        "property_id": u.property_id,
        "where": f"{r.unit_id} · {last_m:%b %Y}",
        "detail": "Tenant has an open lease but didn't submit sales for the month.",
        "impact": est,
        "fix": f"Estimated ${est:,.0f} (last month × the mall's month-over-month change, {mom - 1:+.1%}). Chase the tenant.",
    })

# ---------------------------------------------------------------- traffic checks
traffic = traffic.sort_values(["property_id", "visit_date"]).reset_index(drop=True)
traffic["dow"] = traffic.visit_date.dt.dayofweek
traffic["fixed_visitors"] = traffic.visitors.astype(float)


# Expected visitors: median of the same weekday over the 4 weeks before and after, ignoring zeros
traffic["expected"] = (traffic.visitors.where(traffic.visitors > 0)
                       .groupby([traffic.property_id, traffic.dow])
                       .transform(lambda s: s.rolling(9, center=True, min_periods=3).median()))
traffic["t_ratio"] = traffic.visitors / traffic.expected
zero = traffic.visitors == 0
high = traffic.t_ratio > 1.6
traffic.loc[zero, "fixed_visitors"] = traffic.loc[zero, "expected"].round()
traffic.loc[high, "fixed_visitors"] = (traffic.loc[high, "visitors"] / 2).round()


def runs(mask):
    """Group consecutive flagged days per property into (property, first day, last day, rows)."""
    out = []
    flagged = traffic[mask]
    for p, g in flagged.groupby("property_id"):
        block = (g.visit_date.diff().dt.days != 1).cumsum()
        for _, b in g.groupby(block):
            out.append((p, b.visit_date.min(), b.visit_date.max(), b))
    return out


for p, d0, d1, b in runs(zero):
    issues.append({
        "check": "Zero traffic days",
        "property_id": p,
        "where": f"{d0:%d %b} – {d1:%d %b %Y}",
        "detail": f"Door counters recorded 0 visitors for {len(b)} days in a row while the mall was trading. Counter outage.",
        "impact": round(b.fixed_visitors.sum() - b.visitors.sum()),
        "fix": "Replaced with the median of the same weekday in the surrounding 4 weeks.",
        "unit": "visitors",
    })
for p, d0, d1, b in runs(high):
    issues.append({
        "check": "Traffic spike vs same weekday",
        "property_id": p,
        "where": f"{d0:%d %b} – {d1:%d %b %Y}",
        "detail": f"Visitors about {b.t_ratio.mean():.1f}× the normal level for those weekdays, with no event and no matching sales lift. Counter double-counting.",
        "impact": round(b.fixed_visitors.sum() - b.visitors.sum()),
        "fix": "Halved the counts for those days.",
        "unit": "visitors",
    })

# ---------------------------------------------------------------- monthly roll-ups
traffic["m"] = traffic.visit_date.dt.to_period("M").dt.to_timestamp()
fixed = fixed.sort_values(["property_id", "sales_month", "unit_id"]).reset_index(drop=True)
raw_m = sales.groupby(["property_id", "sales_month"]).reported_sales.sum()
fix_m = fixed.groupby(["property_id", "sales_month"]).fixed_sales.sum()
bud_m = budget.set_index(["property_id", "sales_month"]).budget_sales
tr_raw = traffic.groupby(["property_id", "m"]).visitors.sum()
tr_fix = traffic.groupby(["property_id", "m"]).fixed_visitors.sum()

gla = units.set_index("unit_id").gla_sqft
lease_units = leases.assign(gla=leases.unit_id.map(gla), property_id=leases.unit_id.map(units.set_index("unit_id").property_id))

monthly = {}
for p in props.property_id:
    rec = {k: [] for k in ["actual_raw", "actual_fixed", "budget", "traffic_raw", "traffic_fixed", "occupied_sqft", "leased_sqft"]}
    lp = lease_units[lease_units.property_id == p]
    for m in months:
        me = m + pd.offsets.MonthEnd(0)
        rec["actual_raw"].append(round(float(raw_m.get((p, m), 0)), 2))
        rec["actual_fixed"].append(round(float(fix_m.get((p, m), 0)), 2))
        rec["budget"].append(round(float(bud_m.get((p, m), 0)), 2))
        rec["traffic_raw"].append(int(tr_raw.get((p, m), 0)))
        rec["traffic_fixed"].append(int(tr_fix.get((p, m), 0)))
        rec["occupied_sqft"].append(int(lp[(lp.open_date <= me) & (lp.lease_end >= me)].gla.sum()))
        rec["leased_sqft"].append(int(lp[(lp.lease_start <= me) & (lp.lease_end >= me)].gla.sum()))
    monthly[p] = rec

# ---------------------------------------------------------------- tenant rows for the detail table
names = tenants.set_index("tenant_id")
rows = []
raw_lookup = sales.groupby(key).reported_sales.sum()
for _, r in fixed.iterrows():
    k = (r.unit_id, r.tenant_id, r.sales_month)
    rows.append([
        r.unit_id, r.tenant_id, month_idx[r.sales_month],
        None if pd.isna(r.reported_sales) else round(float(raw_lookup.get(k, 0)), 2),
        round(float(r.fixed_sales), 2), r.flag,
    ])

tenant_info = {t: [names.loc[t, "tenant_name"], names.loc[t, "category"]] for t in tenants.tenant_id}
unit_info = {u.unit_id: [u.property_id, int(u.gla_sqft), u.unit_type] for u in units.itertuples()}

# ---------------------------------------------------------------- lease expiries, next 12 months
t12 = fixed[fixed.sales_month > AS_OF - pd.DateOffset(months=12)].groupby(["unit_id", "tenant_id"]).fixed_sales.sum()
exp = lease_units[(lease_units.lease_end > AS_OF) & (lease_units.lease_end <= AS_OF + pd.DateOffset(months=12))]
expiries = []
for r in exp.sort_values("lease_end").itertuples():
    rent = r.gla * r.base_rent_psf
    s12 = float(t12.get((r.unit_id, r.tenant_id), 0))
    expiries.append({
        "property_id": r.property_id, "unit_id": r.unit_id, "tenant": names.loc[r.tenant_id, "tenant_name"],
        "category": names.loc[r.tenant_id, "category"], "lease_end": f"{r.lease_end:%Y-%m-%d}",
        "gla": int(r.gla), "annual_rent": round(rent), "sales_psf": round(s12 / r.gla, 0) if s12 else None,
        "occ_cost": round(rent / s12, 4) if s12 else None,
    })

out = {
    "as_of": f"{AS_OF:%Y-%m-%d}",
    "months": [f"{m:%Y-%m}" for m in months],
    "properties": [{"id": r.property_id, "name": r.property_name, "city": r.city, "gla": int(r.total_gla_sqft)}
                   for r in props.itertuples()],
    "monthly": monthly,
    "units": unit_info,
    "tenants": tenant_info,
    "sales": rows,
    "issues": issues,
    "expiries": expiries,
}
OUT.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), {len(issues)} issues")
for i in issues:
    print(" -", i["check"], i["property_id"], i["where"], i["impact"])
