"""Supply Chain Risk Radar - Streamlit dashboard around the disruption Random Forest.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from engine import (DEST_PREFIX, ORIGIN_PREFIX, Engine, calendar_alerts, candidate_ports, find_industry_prefix,
                    group_options, haversine_km, routes_for, week_of_year)

ART = Path(__file__).parent / "artifacts"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
KM_TO_MI = 0.621371

st.set_page_config(page_title="Supply Chain Risk Radar", page_icon="🚢", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading model…")
def load_engine() -> Engine | None:
    if not (ART / "model.joblib").exists():
        return None
    model = joblib.load(ART / "model.joblib")
    meta = json.loads((ART / "meta.json").read_text())
    ref = pd.read_csv(ART / "reference.csv.gz")
    return Engine(model, meta, ref)


eng = load_engine()
if eng is None:
    st.title("🚢 Supply Chain Risk Radar")
    st.error("Model artifacts not found. Train and export the model first:")
    st.code('python train_model.py --csv "path/to/model_ready_dataset.csv"', language="bash")
    st.stop()


def fmt(v: float, unit: str) -> str:
    return f"${v:,.0f}" if unit == "USD" else f"{v:,.1f}"


def tier_badge(tier: str) -> str:
    return (f"<span style='background:{C.TIER_COLOR[tier]};color:#0b0b0b;padding:4px 12px;border-radius:6px;"
            f"font-weight:700;font-size:1.1rem'>{C.TIER_ICON[tier]} {tier}</span>")


# ------------------------------------------------------------------ sidebar inputs
st.sidebar.title("🚢 Shipment details")
ship_date = st.sidebar.date_input("Date (ship / arrival week)", value=date.today())

origin_opts = group_options(eng.cols, ORIGIN_PREFIX)
if origin_opts:
    origin_label = st.sidebar.selectbox("Origin region", list(origin_opts))
    origin_col = origin_opts[origin_label]
else:
    origin_label, origin_col = "Unknown origin", None

ind_prefix = find_industry_prefix(eng.cols)
if ind_prefix:
    ind_opts = group_options(eng.cols, ind_prefix)
    industry_label = st.sidebar.selectbox("Industry", list(ind_opts))
    industry_col = ind_opts[industry_label]
else:
    industry_label = st.sidebar.selectbox("Industry", list(C.INDUSTRIES), index=list(C.INDUSTRIES).index(C.DEFAULT_INDUSTRY))
    industry_col = None

route = st.sidebar.selectbox("Shipping route", routes_for(origin_label), format_func=lambda r: r["name"])
state = st.sidebar.selectbox("Destination state", list(C.STATES), format_func=lambda s: C.STATES[s][0],
                             index=list(C.STATES).index("IL"))
cands = candidate_ports(state, route["coasts"])
port = st.sidebar.selectbox("US entry port", cands,
                            format_func=lambda p: f"{p['name']} · {p['km'] * KM_TO_MI:,.0f} mi from {C.STATES[state][0]}")

dest_opts = group_options(eng.cols, DEST_PREFIX)
dest_col = next((c for lbl, c in dest_opts.items() if "north america" in lbl.lower()), None)

lane, fell_back = eng.lane(origin_col, dest_col, industry_col)
woy = week_of_year(ship_date)
baseline = eng.seasonal_baseline(lane, woy)
lane_key = f"{origin_col}|{industry_col}|{woy}"

with st.sidebar.expander("Current conditions", expanded=True):
    st.caption("Defaults are the typical values for this lane and time of year. Move a slider to reflect what you see now.")
    slider_vals: dict[str, float] = {}
    for k in C.EDITABLE_DRIVERS:
        if k not in eng.drivers:
            continue
        label, unit, _ = eng.drivers[k]
        lo, hi = (float(x) for x in eng.ref[k].quantile([0.005, 0.995]))
        default = float(np.clip(baseline[k], lo, hi))
        slider_vals[k] = st.slider(f"{label} ({unit})", lo, hi, default, step=(hi - lo) / 200,
                                   format="%.1f", key=f"{k}|{lane_key}")

with st.sidebar.expander("Port-specific congestion (optional)"):
    st.caption("Enter +/- index points for any port to reflect what you know about it (e.g. a backlog). "
               "The map and the Port alternatives tab re-score each port with this.")
    offs_df = pd.DataFrame({"Port": [p[0] for p in C.PORTS], "Congestion adj. (+/- pts)": 0.0})
    offs_df = st.data_editor(offs_df, hide_index=True, disabled=["Port"], height=300, key="port_offsets")
port_offsets = dict(zip(offs_df["Port"], offs_df["Congestion adj. (+/- pts)"].fillna(0.0)))

with st.sidebar.expander("Advanced"):
    persist = st.checkbox("Assume these conditions held over the last few weeks", value=True,
                          help="Shifts the lag / rolling-window features by the same amount as your slider change.")
    start_override = None
    if "weeks_since_start" in eng.cols:
        start_override = st.date_input("Date of week 0 in your dataset", value=eng.start_date(),
                                       help="The CSV only has year and weeks_since_start, so week 0 is assumed to be 1 Jan of the first year.")

start = eng.start_date(start_override)

# ------------------------------------------------------------------ scoring
base_overrides = dict(slider_vals)
cong_slider = base_overrides.get("port_congestion_index")
sel_overrides = dict(base_overrides)
if cong_slider is not None:
    sel_overrides["port_congestion_index"] = cong_slider + port_offsets.get(port["name"], 0.0)

row_base = eng.build_row(baseline, ship_date, start, origin_col, dest_col, industry_col, base_overrides, persist)
row = eng.build_row(baseline, ship_date, start, origin_col, dest_col, industry_col, sel_overrides, persist)
res = eng.analyse(row, baseline, lane, industry_label)
p, tier = res["p"], res["tier"]
p_typical = float(eng.score_rows([eng.build_row(baseline, ship_date, start, origin_col, dest_col, industry_col, {}, persist)])[0])

slat, slon = C.STATES[state][1], C.STATES[state][2]
ports_all = [{"name": n, "lat": la, "lon": lo, "coast": co, "hurricane": hu,
              "km": haversine_km(slat, slon, la, lo)} for n, la, lo, co, hu in C.PORTS]
cong_by_port = {q["name"]: (cong_slider or float(row_base.get("port_congestion_index", 0))) + port_offsets.get(q["name"], 0.0)
                for q in ports_all}
port_scores = eng.score_ports(row_base, baseline, ports_all, cong_by_port, persist)
port_scores[port["name"]] = p  # selected port uses the exact scenario row

flagged = [d for d in res["drivers"] if d["flagged"]]
cal = calendar_alerts(ship_date, origin_label, route, port)

# ------------------------------------------------------------------ header + KPIs
st.title("Supply Chain Risk Radar")
st.caption(f"{industry_label} · {origin_label} → {C.STATES[state][0]} via {port['name']} · {route['name']} · week of {ship_date:%b %d, %Y}")

if eng.beyond_training(ship_date, start):
    st.warning("This date is later than the last week in the training data. `year` / `weeks_since_start` are outside "
               "the range the model has seen, so it treats this week like the latest period it knows. "
               "Retraining with `--drop-time-features` avoids this.")
if fell_back:
    st.info("Few historical rows match this exact lane, so baselines use a broader slice of the data.")

banner = {
    "High": st.error, "Elevated": st.warning, "Moderate": st.info, "Low": st.success,
}[tier]
n_alerts = len([d for d in flagged if d["alert"] != "Normal"])
banner(f"{C.TIER_ICON[tier]} **{tier} disruption risk** — modelled probability of a disruption next week is "
       f"**{p:.1%}** ({p / eng.base_rate:.1f}× the {eng.base_rate:.1%} historical average). "
       + (f"{len(flagged)} risk factor{'s' if len(flagged) != 1 else ''} flagged." if flagged else "No risk factors flagged."))

k1, k2, k3, k4 = st.columns(4)
k1.markdown("**Risk tier**")
k1.markdown(tier_badge(tier), unsafe_allow_html=True)
k2.metric("Disruption probability", f"{p:.1%}", f"{(p - p_typical) * 100:+.1f} pp vs typical week", delta_color="inverse",
          help=f"A typical week for this lane and season scores {p_typical:.1%}. The all-data average disruption rate is {eng.base_rate:.1%}.")
k3.metric("Percentile vs. history", f"{res['pct']:.0f}th", help="Where this probability falls among held-out test weeks.")
k4.metric("Flagged risk factors", len(flagged), f"{len(cal)} seasonal / route alerts", delta_color="off")
hit = eng.meta["tier_hit_rates"].get(tier)
st.caption(f"In held-out test data, {hit:.1%} of weeks scored in the **{tier}** tier were followed by a disruption "
           f"(overall {eng.base_rate:.1%}). Model test ROC-AUC is {eng.meta['metrics']['test_auc']:.2f}, so treat this as a "
           "screening signal, not a forecast.")

# ------------------------------------------------------------------ map + driver chart
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Route & port risk map")
    best_alt = min((q for q in ports_all if q["name"] != port["name"]), key=lambda q: port_scores[q["name"]])
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=[state], z=[1], locationmode="USA-states", colorscale=[[0, "#cde2fb"], [1, "#cde2fb"]],
        showscale=False, marker_line_color="#256abf", marker_line_width=1.5, name="Destination state",
        hovertemplate=f"Destination: {C.STATES[state][0]}<extra></extra>", showlegend=True))
    fig.add_trace(go.Scattergeo(lon=[slon, port["lon"]], lat=[slat, port["lat"]], mode="lines",
                                line=dict(width=2, dash="dot", color=INK2), hoverinfo="skip", showlegend=False))
    symbols = {"Low": "circle", "Moderate": "triangle-up", "Elevated": "diamond", "High": "square"}
    by_tier: dict[str, list] = {t[0]: [] for t in C.TIERS}
    for q in ports_all:
        by_tier[eng.tier(port_scores[q["name"]])].append(q)
    for t, _, col in C.TIERS:
        qs = by_tier[t]
        if not qs:
            continue
        fig.add_trace(go.Scattergeo(
            lon=[q["lon"] for q in qs], lat=[q["lat"] for q in qs], mode="markers", name=f"{C.TIER_ICON[t]} {t}",
            marker=dict(size=15, color=col, symbol=symbols[t], line=dict(width=2, color=SURFACE)),
            text=[f"<b>{q['name']}</b><br>{q['coast']} coast<br>Risk: {port_scores[q['name']]:.1%} ({t})"
                  f"<br>Congestion used: {cong_by_port[q['name']]:.1f}<br>{q['km'] * KM_TO_MI:,.0f} mi to {state}"
                  for q in qs], hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scattergeo(
        lon=[port["lon"]], lat=[port["lat"]], mode="markers+text", name="Selected entry port",
        marker=dict(size=30, symbol="circle-open", line=dict(width=3, color=INK)),
        text=[port["name"]], textposition="bottom center", textfont=dict(color=INK, size=12), hoverinfo="skip"))
    if best_alt["name"] != port["name"] and port_scores[best_alt["name"]] < p - 1e-4:
        fig.add_trace(go.Scattergeo(
            lon=[best_alt["lon"]], lat=[best_alt["lat"]], mode="markers+text", name="Lowest-risk alternative",
            marker=dict(size=24, symbol="circle-open", line=dict(width=3, color=BLUE)),
            text=[best_alt["name"]], textposition="top center", textfont=dict(color=BLUE, size=12), hoverinfo="skip"))
    fig.update_geos(scope="usa", projection_type="albers usa", showlakes=True, lakecolor=SURFACE, landcolor="#f0efec",
                    bgcolor=SURFACE, subunitcolor="#c3c2b7", showsubunits=True)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(color=INK)), font=dict(color=INK))
    st.plotly_chart(fig, width="stretch")
    st.caption("Port markers use shape and label as well as colour. Every port is scored with the same conditions "
               "unless you enter port-specific congestion in the sidebar.")

with right:
    st.subheader("What is driving the risk")
    drv = sorted(res["drivers"], key=lambda d: d["contribution"])
    fig2 = go.Figure(go.Bar(
        x=[d["contribution"] * 100 for d in drv], y=[d["label"] for d in drv], orientation="h",
        marker=dict(color=[BLUE if d["contribution"] > 0 else "#b8b7ae" for d in drv]),
        text=[f"{d['contribution'] * 100:+.2f} pp" for d in drv], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}<br>%{x:+.2f} percentage points of risk<extra></extra>"))
    fig2.update_layout(height=520, margin=dict(l=0, r=60, t=10, b=40), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                       xaxis=dict(title="Risk added vs. a normal-conditions week (percentage points)", gridcolor="#e6e5e0",
                                  zeroline=True, zerolinecolor=INK2, title_font=dict(color=INK2, size=12), tickfont=dict(color=INK2)),
                       yaxis=dict(tickfont=dict(color=INK)), bargap=0.45, font=dict(color=INK))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Each bar = modelled risk with that driver at today's level minus the risk with it at its seasonal norm.")

# ------------------------------------------------------------------ tabs
tab_flags, tab_fix, tab_ports, tab_about = st.tabs(["🚩 Flagged risks", "🛠 Workarounds", "📍 Port alternatives", "ℹ️ About the model"])

with tab_flags:
    status_txt = {"Critical": f"{C.TIER_ICON['High']} Critical", "High": f"{C.TIER_ICON['Elevated']} High",
                  "Normal": f"{C.TIER_ICON['Low']} OK"}
    rows = []
    for d in res["drivers"]:
        s = status_txt[d["alert"]]
        if d["alert"] == "Normal" and d["model_driven"]:
            s = f"{C.TIER_ICON['Moderate']} Model-flagged"
        rows.append({"Status": s, "Driver": d["label"], "Current": fmt(d["current"], d["unit"]),
                     "Typical (this season)": fmt(d["typical"], d["unit"]),
                     "Percentile (worse = higher)": round(d["worse_pct"]),
                     "Model contribution (pp)": round(d["contribution"] * 100, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Percentile (worse = higher)": st.column_config.ProgressColumn(
                     "Percentile (worse = higher)", min_value=0, max_value=100, format="%d")})
    st.caption(f"**Condition alerts** fire when a driver is in the worse tail of this lane's history "
               f"(industry-tuned: {industry_label} flags earlier on the drivers it is most sensitive to). "
               "**Model-flagged** means the Random Forest attributes a meaningful share of the risk to that driver. "
               "Container availability is assumed to be worse when *low*; change this in `config.py` if your index runs the other way.")
    st.markdown("#### Seasonal & route alerts")
    if not cal:
        st.write("None for this date and route.")
    for a in cal:
        (st.warning if a["level"] == "warn" else st.info)(f"**{a['title']}** — {a['detail']}")
    st.caption("Seasonal and route alerts are rules of thumb, not model output.")

with tab_fix:
    if not flagged and tier in ("Low", "Moderate"):
        st.success("Nothing flagged. Standard monitoring is enough: keep normal safety stock and watch carrier ETAs.")
    if tier in ("High", "Elevated"):
        st.markdown("#### Overall contingency steps")
        for s in C.GENERIC_HIGH_RISK:
            st.markdown(f"- {s}")
    for d in flagged[:6]:
        eff = d["contribution"] * 100
        title = f"{d['label']} — {d['alert'] if d['alert'] != 'Normal' else 'Model-flagged'}"
        with st.expander(title, expanded=d["alert"] == "Critical" or d is flagged[0]):
            st.write(f"Now **{fmt(d['current'], d['unit'])}** vs typical **{fmt(d['typical'], d['unit'])}** "
                     f"({d['worse_pct']:.0f}th percentile on the worse side).")
            if eff > 0:
                st.write(f"Returning this to its seasonal norm would lower modelled risk by about **{eff:.2f} pp**.")
            for s in C.WORKAROUNDS.get(d["key"], []):
                st.markdown(f"- {s}")
    ctag = [t for t in route["chokepoints"] if t in C.CHOKEPOINT_NOTES]
    if ctag and any(d["key"] == "geopolitical_risk_score" for d in flagged):
        st.warning("Your route passes through an exposed chokepoint while geopolitical risk is elevated: "
                   + " ".join(C.CHOKEPOINT_NOTES[t] for t in ctag))
    extras = res["profile"]["extra"]
    if extras and (flagged or tier != "Low"):
        st.markdown(f"#### Specific to {industry_label}")
        for s in extras:
            st.markdown(f"- {s}")
    if best_alt["name"] != port["name"] and port_scores[best_alt["name"]] < p - 0.002:
        st.markdown("#### Alternative port")
        st.markdown(f"- **{best_alt['name']}** scores {port_scores[best_alt['name']]:.1%} vs {p:.1%} here "
                    f"({(p - port_scores[best_alt['name']]) * 100:.2f} pp lower).")
    if flagged:
        fixed_row = eng.neutralize(row, baseline, [d["key"] for d in flagged])
        p_fixed = float(eng.score_rows([fixed_row])[0])
        st.markdown("#### If all flagged factors returned to normal")
        st.metric("Modelled risk", f"{p_fixed:.1%}", f"{(p_fixed - p) * 100:+.2f} pp vs now", delta_color="inverse")
        st.caption("What-if based on historical associations in the data; it is not proof that a workaround causes this reduction.")

with tab_ports:
    only_route = st.checkbox("Only ports compatible with this route", value=True)
    allowed = {c["name"] for c in cands} if only_route else {q["name"] for q in ports_all}
    rows = []
    for q in sorted(ports_all, key=lambda q: port_scores[q["name"]]):
        if q["name"] not in allowed:
            continue
        sp = port_scores[q["name"]]
        rows.append({"Port": q["name"] + (" (selected)" if q["name"] == port["name"] else ""), "Coast": q["coast"],
                     "Risk": round(sp * 100, 2), "Tier": f"{C.TIER_ICON[eng.tier(sp)]} {eng.tier(sp)}",
                     "vs selected (pp)": round((sp - p) * 100, 2), "Congestion used": round(cong_by_port[q["name"]], 1),
                     f"Miles to {state}": round(q["km"] * KM_TO_MI)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Risk": st.column_config.NumberColumn("Risk (%)", format="%.2f")})
    if not any(abs(v) > 0 for v in port_offsets.values()):
        st.info("All ports currently use the same conditions, so they score alike. Enter port-specific congestion "
                "adjustments in the sidebar to compare real alternatives. The distance column shows the inland trade-off.")
    st.caption("Straight-line distance from the destination state's centre, not road or rail miles.")

with tab_about:
    m = eng.meta["metrics"]
    a, b, c_ = st.columns(3)
    a.metric("In-sample ROC-AUC", f"{m['train_auc']:.3f}")
    b.metric("Out-of-sample ROC-AUC", f"{m['test_auc']:.3f}")
    c_.metric("Disruption base rate", f"{m['base_rate']:.1%}")
    if m["train_auc"] - m["test_auc"] > 0.15:
        st.warning(f"The model fits its training data almost perfectly ({m['train_auc']:.2f}) but generalises weakly "
                   f"({m['test_auc']:.2f}). That is the signature of overfitting, so the dashboard uses *relative* tiers "
                   "(percentiles of held-out predictions) rather than treating the raw probability as exact.")
    if not eng.meta["dropped_time_features"] and "weeks_since_start" in eng.cols:
        st.warning("`weeks_since_start` and `year` were the model's top features. They let it memorise *when* disruptions "
                   "happened, which does not carry over to new dates. Try `python train_model.py --drop-time-features`.")
    st.markdown(f"""
**How the dashboard works**
- Your inputs pick a lane (origin region → North America) and a typical week for that time of year from the dataset.
- Sliders override the current conditions; lag and rolling features shift with them if the persistence option is on.
- The Random Forest scores the scenario. Tiers are percentiles of out-of-sample predictions:
  Low < {eng.thr['moderate']:.1%} ≤ Moderate < {eng.thr['elevated']:.1%} ≤ Elevated < {eng.thr['high']:.1%} ≤ High.
- Driver contributions come from swapping one driver (and its lags) back to its seasonal norm and re-scoring.

**Assumptions to be aware of**
- Industry, route and port geography are not model inputs unless your CSV has matching columns
  ({'industry columns detected, so industry is a model input' if ind_prefix else 'no industry columns found, so industry only tunes alert sensitivity and the playbook'}).
- Workarounds and seasonal alerts are rules of thumb from `config.py`, not learned from your data.
- Training rows: {eng.meta['n_rows']:,}; features used: {len(eng.features)}.
""")
