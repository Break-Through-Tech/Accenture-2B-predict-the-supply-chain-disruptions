"""Scenario building, scoring, risk flags and what-if analysis around the Random Forest."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config as C

ORIGIN_PREFIX = "origin_region_"
DEST_PREFIX = "destination_region_"
INDUSTRY_PREFIXES = ("industry_", "sector_", "product_category_", "commodity_category_", "commodity_type_")


# ------------------------------------------------------------------ helpers
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def group_options(columns, prefix) -> dict[str, str]:
    """{label: column} for every one-hot column starting with prefix."""
    return {c[len(prefix):]: c for c in columns if c.startswith(prefix)}


def find_industry_prefix(columns) -> str | None:
    for p in INDUSTRY_PREFIXES:
        if sum(c.startswith(p) for c in columns) >= 2:
            return p
    return None


def driver_cols(columns, base: str) -> list[str]:
    return [c for c in columns if c == base or c.startswith(base + "_lag") or c.startswith(base + "_rolling")]


def match_industry_profile(label: str) -> dict:
    if label in C.INDUSTRIES:
        return C.INDUSTRIES[label]
    low = label.lower()
    for prof in C.INDUSTRIES.values():
        if any(k in low for k in prof["keywords"]):
            return prof
    return C.INDUSTRIES[C.DEFAULT_INDUSTRY]


def routes_for(origin_label: str) -> list[dict]:
    low = origin_label.lower()
    for key, routes in C.ROUTES.items():
        if key in low:
            return routes
    return C.DEFAULT_ROUTES


def candidate_ports(state_code: str, coasts: list[str]):
    """Ports sorted by distance from the destination state's centroid."""
    _, slat, slon = C.STATES[state_code]
    out = []
    for name, lat, lon, coast, hurricane in C.PORTS:
        if "Any" not in coasts and coast not in coasts:
            continue
        out.append({"name": name, "lat": lat, "lon": lon, "coast": coast, "hurricane": hurricane,
                    "km": haversine_km(slat, slon, lat, lon)})
    out.sort(key=lambda p: p["km"])
    return out


def week_of_year(d: date) -> int:
    return min((d.timetuple().tm_yday - 1) // 7, 51)


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, model, meta: dict, ref: pd.DataFrame):
        self.model = model
        self.model.n_jobs = 1  # tiny batches: threading overhead costs more than it saves
        self.meta = meta
        self.ref = ref
        self.features: list[str] = meta["feature_names"]
        self.cols: list[str] = meta["all_columns"]
        self.target = meta["target"]
        self.thr = meta["thresholds"]
        self.q = np.array(meta["prob_quantiles"])
        self.base_rate = meta["metrics"]["base_rate"]
        self.binary_cols = [c for c in self.cols if ref[c].isin([0, 1]).all()]
        self.drivers = {k: v for k, v in C.DRIVERS.items() if k in self.cols}

    # ---------------------------------------------------------- data slices
    def lane(self, origin_col=None, dest_col=None, industry_col=None):
        m = pd.Series(True, index=self.ref.index)
        for c in (origin_col, dest_col, industry_col):
            if c and c in self.ref:
                m &= self.ref[c] == 1
        if m.sum() >= 100:
            return self.ref[m], False
        # too few rows for this exact lane: relax industry, then destination
        for drop in ((industry_col,), (industry_col, dest_col)):
            m = pd.Series(True, index=self.ref.index)
            for c in (origin_col, dest_col, industry_col):
                if c and c not in drop and c in self.ref:
                    m &= self.ref[c] == 1
            if m.sum() >= 100:
                return self.ref[m], True
        return self.ref, True

    def seasonal_baseline(self, lane: pd.DataFrame, woy: int, win: int = 4) -> pd.Series:
        sub = lane
        if "weeks_since_start" in lane:
            d = np.abs((lane["weeks_since_start"] % 52) - woy)
            d = np.minimum(d, 52 - d)
            seasonal = lane[d <= win]
            if len(seasonal) >= 30:
                sub = seasonal
        base = sub[self.cols].median()
        for c in self.binary_cols:
            base[c] = float(sub[c].mean() >= 0.5)
        return base

    # ---------------------------------------------------------- date mapping
    def start_date(self, override: date | None = None) -> date:
        if override:
            return override
        return date(int(self.meta.get("start_year") or 2020), 1, 1)

    def beyond_training(self, d: date, start: date) -> bool:
        mx = self.meta.get("max_weeks_since_start")
        if mx is None or "weeks_since_start" not in self.cols:
            return False
        return (d - start).days / 7 > mx + 1

    def _date_features(self, d: date, start: date) -> dict:
        weeks = (d - start).days / 7
        woy = week_of_year(d)
        f = {
            "weeks_since_start": weeks, "year": d.year, "month": d.month, "quarter": (d.month - 1) // 3 + 1,
            "week": woy + 1, "week_of_year": woy + 1, "weekofyear": woy + 1,
            "day_of_year": d.timetuple().tm_yday,
            "month_sin": math.sin(2 * math.pi * d.month / 12), "month_cos": math.cos(2 * math.pi * d.month / 12),
            "week_sin": math.sin(2 * math.pi * (woy + 1) / 52), "week_cos": math.cos(2 * math.pi * (woy + 1) / 52),
        }
        return {k: v for k, v in f.items() if k in self.cols}

    # ---------------------------------------------------------- scenario row
    def _derived(self, row: pd.Series) -> None:
        if "congestion_x_delay" in row.index and {"port_congestion_index", "shipping_delay_days"} <= set(row.index):
            row["congestion_x_delay"] = row["port_congestion_index"] * row["shipping_delay_days"]

    def build_row(self, baseline: pd.Series, d: date, start: date, origin_col, dest_col, industry_col,
                  overrides: dict[str, float], persist: bool = True) -> pd.Series:
        row = baseline.copy()
        for k, v in self._date_features(d, start).items():
            row[k] = v
        for prefix, chosen in ((ORIGIN_PREFIX, origin_col), (DEST_PREFIX, dest_col)):
            if chosen:
                for c in self.cols:
                    if c.startswith(prefix):
                        row[c] = 1.0 if c == chosen else 0.0
        ip = find_industry_prefix(self.cols)
        if ip and industry_col:
            for c in self.cols:
                if c.startswith(ip):
                    row[c] = 1.0 if c == industry_col else 0.0
        for base, val in overrides.items():
            if base not in row.index:
                continue
            delta = float(val) - float(baseline[base])
            row[base] = float(val)
            if persist and delta != 0:
                for c in driver_cols(self.cols, base):
                    if c != base:
                        row[c] = float(baseline[c]) + delta
        self._derived(row)
        return row

    # ---------------------------------------------------------- scoring
    def score_rows(self, rows: list[pd.Series]) -> np.ndarray:
        df = pd.DataFrame([r[self.features] for r in rows], columns=self.features)
        return self.model.predict_proba(df)[:, 1]

    def tier(self, p: float) -> str:
        if p >= self.thr["high"]:
            return "High"
        if p >= self.thr["elevated"]:
            return "Elevated"
        if p >= self.thr["moderate"]:
            return "Moderate"
        return "Low"

    def percentile(self, p: float) -> float:
        return float(np.clip(np.searchsorted(self.q, p, side="right") - 1, 0, 100))

    # ---------------------------------------------------------- what-ifs
    def neutralize(self, row: pd.Series, baseline: pd.Series, bases: list[str]) -> pd.Series:
        r = row.copy()
        for b in bases:
            for c in driver_cols(self.cols, b):
                r[c] = baseline[c]
        self._derived(r)
        return r

    def analyse(self, row: pd.Series, baseline: pd.Series, lane: pd.DataFrame, industry_label: str) -> dict:
        """Score the scenario, attribute risk to drivers, and build the flag table."""
        keys = list(self.drivers)
        rows = [row] + [self.neutralize(row, baseline, [k]) for k in keys] + [self.neutralize(row, baseline, keys)]
        probs = self.score_rows(rows)
        p = float(probs[0])
        contrib = {k: p - float(probs[i + 1]) for i, k in enumerate(keys)}
        p_all_fixed = float(probs[-1])

        prof = match_industry_profile(industry_label)
        recs = []
        for k in keys:
            label, unit, direction = self.drivers[k]
            cur = float(row[k])
            typical = float(baseline[k])
            hist = lane[k].to_numpy()
            pct = float((hist <= cur).mean() * 100)
            worse = pct if direction > 0 else 100 - pct
            w = prof["weights"].get(k, 1.0)
            thr = float(np.clip(80 / w, 55, 95))
            alert = "Critical" if worse >= max(95, thr) else ("High" if worse >= thr else "Normal")
            model_driven = contrib[k] >= max(0.10 * p, 0.002)
            recs.append({
                "key": k, "label": label, "unit": unit, "current": cur, "typical": typical,
                "percentile": pct, "worse_pct": worse, "weight": w, "alert": alert,
                "contribution": contrib[k], "model_driven": model_driven,
                "flagged": alert != "Normal" or model_driven,
            })
        recs.sort(key=lambda r: (r["alert"] == "Critical", r["worse_pct"] * r["weight"], r["contribution"]), reverse=True)
        return {"p": p, "tier": self.tier(p), "pct": self.percentile(p), "drivers": recs,
                "p_all_fixed": p_all_fixed, "profile": prof}

    # ---------------------------------------------------------- multi-port
    def score_ports(self, row: pd.Series, baseline: pd.Series, ports: list[dict], congestion_by_port: dict,
                    persist: bool = True) -> dict[str, float]:
        rows = []
        for p in ports:
            r = row.copy()
            cong = congestion_by_port.get(p["name"], row["port_congestion_index"])
            if "port_congestion_index" in r.index:
                delta = float(cong) - float(baseline["port_congestion_index"])
                r["port_congestion_index"] = float(cong)
                if persist:
                    for c in driver_cols(self.cols, "port_congestion_index"):
                        if c != "port_congestion_index":
                            r[c] = float(baseline[c]) + delta
                self._derived(r)
            rows.append(r)
        probs = self.score_rows(rows)
        return {p["name"]: float(pr) for p, pr in zip(ports, probs)}


# ------------------------------------------------------------------ calendar heuristics
def calendar_alerts(d: date, origin_label: str, route: dict, port: dict) -> list[dict]:
    """Seasonal rules of thumb. These are NOT model output."""
    out = []
    m = d.month
    if port["hurricane"] and 6 <= m <= 11:
        peak = 8 <= m <= 10
        out.append({"title": "Atlantic hurricane season",
                    "detail": f"{port['name']} is hurricane-exposed and {d:%b} is "
                              f"{'peak' if peak else 'within'} season (Jun–Nov): expect possible port closures and vessel diversions.",
                    "level": "warn" if peak else "info"})
    if (m in (12, 1, 2, 3) and port["coast"] in ("East", "Gulf")) or (m in (12, 1, 2) and port["coast"] == "West"):
        out.append({"title": "Winter weather window",
                    "detail": "Winter storms can delay inland rail and trucking legs and slow terminal operations.", "level": "info"})
    if "asia" in origin_label.lower():
        y = d.year
        for yr in (y, y + 1):
            if yr in C.LUNAR_NEW_YEAR:
                lny = date(yr, *C.LUNAR_NEW_YEAR[yr])
                if lny - timedelta(days=21) <= d <= lny + timedelta(days=28):
                    out.append({"title": "Lunar New Year shutdown window",
                                "detail": f"Factories slow or close around {lny:%b %d}; expect blank sailings and "
                                          "a rebound surge in the weeks after.", "level": "warn"})
                    break
        if m == 10 and d.day <= 7:
            out.append({"title": "China Golden Week", "detail": "Oct 1–7 factory and trucking slowdowns at origin.", "level": "info"})
    if (m == 7 and d.day >= 15) or m in (8, 9, 10):
        out.append({"title": "Retail peak season", "detail": "Aug–Oct is peak import season: tight capacity, surcharges and terminal congestion.", "level": "info"})
    if (m == 11 and d.day >= 15) or m == 12:
        out.append({"title": "Holiday / Cyber-week volume", "detail": "Late-year e-commerce volume strains drayage, warehouses and last-mile.", "level": "info"})
    for tag in route["chokepoints"]:
        if tag == "panama" and (m in (12, 1, 2, 3, 4)):
            out.append({"title": "Panama Canal dry season", "detail": C.CHOKEPOINT_NOTES["panama"], "level": "warn"})
        elif tag in C.CHOKEPOINT_NOTES and tag != "panama":
            out.append({"title": "Route exposure: " + tag.replace("_", " ").title(), "detail": C.CHOKEPOINT_NOTES[tag], "level": "info"})
    return out
