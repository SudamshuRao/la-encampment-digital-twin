"""
LA Encampment Risk Digital Twin -- SNN-2
Zoomable map of LA hex grid -> click (or pick) a hex -> adjust its
contextual variables with sliders -> see SNN-2's predicted encampment
risk update live, decomposed into main-effect AND interaction-effect
contributions. Presentation-friendly layout: map + risk + headline
interaction always visible up top, everything else organized into tabs
instead of one long scroll.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from jinja2 import Template as _JinjaTemplate
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

from snn2_inference import SNN2Model
from hexgeom import hexes_to_geojson
from labels2 import (
    FEATURE_LABELS_L2, FEATURE_GROUPS_L2, ACTIONABLE_GROUPS_L2,
    GROUP_ORDER_L2, RESOLUTION_LABELS, interaction_label,
)
from optimizer2 import greedy_optimize_snn2, is_actionable_step_l2, actionable_subset_risk_l2
from scenarios import run_population_optimization

st.set_page_config(page_title="LA Encampment Risk Digital Twin", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESOLUTIONS = ["quarter", "half", "three_fourth"]


@st.cache_resource
def load_model(res):
    return SNN2Model(os.path.join(MODEL_DIR, f"snn2_{res}.json"))


@st.cache_data
def load_hexdf(res):
    return pd.read_csv(os.path.join(DATA_DIR, f"hex_lookup_{res}.csv"))


@st.cache_data
def load_feature_summary(res):
    return pd.read_csv(os.path.join(MODEL_DIR, f"snn2_feature_summary_{res}.csv")).set_index("feature")


RISK_COLORMAP = cm.LinearColormap(colors=["#1a9850", "#fee08b", "#d73027"], vmin=0.0, vmax=1.0)


def _legend_bottom_left(colormap: cm.ColorMap) -> cm.ColorMap:
    src = colormap._template.environment.loader.get_source(
        colormap._template.environment, "color_scale.js")[0]
    colormap._template = _JinjaTemplate(src.replace("position: 'topright'", "position: 'bottomleft'"))
    return colormap


def risk_color(risk, rmin, rmax):
    t = 0.0 if rmax <= rmin else (risk - rmin) / (rmax - rmin)
    return RISK_COLORMAP(min(max(t, 0.0), 1.0))


def build_map(hex_df, selected_grid_id, current_risk_override=None, basemap="light", height=520):
    rmin, rmax = hex_df["baseline_risk"].quantile([0.05, 0.95])
    center_lat, center_lon = hex_df["centroid_lat"].mean(), hex_df["centroid_lon"].mean()

    tiles = "cartodbdark_matter" if basemap == "dark" else "cartodbpositron"
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=tiles)
    Fullscreen(position="topright").add_to(m)

    display_df = hex_df.copy()
    if current_risk_override is not None and selected_grid_id is not None:
        display_df.loc[display_df["GRID_ID"] == selected_grid_id, "baseline_risk"] = current_risk_override

    gj = hexes_to_geojson(display_df, risk_col="baseline_risk")
    quiet_border = "#ffffff" if basemap == "dark" else "#4d4d4d"
    selected_border = "#00e5ff"

    def style_fn(feature):
        gid = feature["properties"]["GRID_ID"]
        risk = feature["properties"]["risk"]
        is_selected = gid == selected_grid_id
        return {
            "fillColor": risk_color(risk, rmin, rmax),
            "color": selected_border if is_selected else quiet_border,
            "weight": 3.5 if is_selected else 0.25,
            "fillOpacity": 0.88 if is_selected else 0.72,
            "opacity": 1.0 if is_selected else 0.35,
        }

    folium.GeoJson(
        gj, style_function=style_fn,
        highlight_function=lambda f: {"weight": 2.5, "color": selected_border},
        tooltip=folium.GeoJsonTooltip(
            fields=["GRID_ID", "risk", "top_interaction"],
            aliases=["Hex", "Predicted risk", "Top interaction"],
            style=("background-color: white; color: #333; font-family: sans-serif; "
                   "font-size: 13px; padding: 6px; border-radius: 4px;"),
        ),
        name="hexes",
    ).add_to(m)

    legend = RISK_COLORMAP.scale(rmin, rmax)
    legend.caption = "Predicted encampment risk"
    legend = _legend_bottom_left(legend)
    legend.add_to(m)
    return m


def render_sliders(model, feat_summary, res, gid):
    for group in GROUP_ORDER_L2:
        group_feats = [f for f in model.base_features if FEATURE_GROUPS_L2.get(f) == group]
        if not group_feats:
            continue
        with st.expander(f"{group} ({len(group_feats)})", expanded=(group == "Facilities / Services")):
            for i in range(0, len(group_feats), 2):
                pair = group_feats[i:i + 2]
                cols = st.columns(len(pair))
                for col, f in zip(cols, pair):
                    if f not in feat_summary.index:
                        continue
                    row = feat_summary.loc[f]
                    lo, hi = float(row["raw_p05"]), float(row["raw_p95"])
                    if hi <= lo:
                        lo, hi = float(row["raw_min"]), float(row["raw_max"])
                    if hi <= lo:
                        hi = lo + 1.0
                    step = max((hi - lo) / 100, 0.01)
                    val = float(np.clip(st.session_state.slider_values[f], lo, hi))
                    with col:
                        new_val = st.slider(
                            FEATURE_LABELS_L2.get(f, f), min_value=lo, max_value=hi,
                            value=val, step=step, key=f"slider_{res}_{gid}_{f}",
                        )
                    st.session_state.slider_values[f] = new_val


def render_contributions(model, contribs):
    main_contribs = {f: v for f, v in contribs.items() if f in model.base_features}
    inter_contribs = {f: v for f, v in contribs.items() if f not in model.base_features}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Main-effect contributions**")
        df = pd.DataFrame({
            "feature": [FEATURE_LABELS_L2.get(f, f) for f in main_contribs],
            "contribution": list(main_contribs.values()),
        }).sort_values("contribution")
        st.bar_chart(df.set_index("feature"), height=420)
    with c2:
        st.markdown("**Interaction-effect contributions**")
        st.caption("Pairwise terms selected during training (top-20 by RF importance).")
        inter_labels = {name: interaction_label(*name.split("__x__")) for name in inter_contribs}
        df2 = pd.DataFrame({
            "interaction": [inter_labels[f] for f in inter_contribs],
            "contribution": list(inter_contribs.values()),
        }).sort_values("contribution")
        st.bar_chart(df2.set_index("interaction"), height=420)


def render_optimizer(model, feat_summary, res, gid):
    opt_mode = st.radio(
        "Mode", ["actionable", "exploratory", "combined"],
        format_func=lambda m: {
            "actionable": "Actionable only (increase-only)",
            "exploratory": "Exploratory (increase or decrease)",
            "combined": "Combined — both directions, tagged",
        }[m],
        key=f"optmode_{res}_{gid}", horizontal=True,
    )
    st.caption(
        "All three modes now respect the project's frozen-variable rule: only "
        "Facilities/Services variables are ever touched. Socioeconomic, "
        "demographic, crime, hazard, roadway, hydrology, and land-cover "
        "variables are never candidates, in any mode."
    )

    captions = {
        "actionable": "Greedy search over realistic +1 increments only (never decreases, "
                       "never exceeds the 95th-percentile observed value). Safe to present "
                       "as real intervention ideas.",
        "exploratory": "⚠️ Allows +1 or -1 moves. A decrease (e.g. fewer libraries) isn't a "
                        "realistic intervention either — this shows what the fitted surface "
                        "says minimizes risk mathematically within the non-frozen set, not a "
                        "real-world action list.",
        "combined": "Same non-frozen variable set and both directions as Exploratory, but "
                    "every step is tagged ✅ Actionable (an increase) or ⚠️ Correlational "
                    "(a decrease), with a separate re-scored 'actionable-only' risk. A shared "
                    "K budget means large decreases can crowd out realistic increases.",
    }
    st.caption(captions[opt_mode])

    k = st.slider("Max interventions (K)", 1, 10, 5, key=f"k_{res}_{gid}")

    candidates = [f for f in model.base_features if FEATURE_GROUPS_L2.get(f) in ACTIONABLE_GROUPS_L2
                  and f in feat_summary.index]

    if st.button("Run optimizer", type="primary"):
        if opt_mode == "actionable":
            bounds = {f: (0.0, float(feat_summary.loc[f, "raw_p95"])) for f in candidates}
            allow_decrease = False
        else:
            bounds = {f: (float(feat_summary.loc[f, "raw_p05"]), float(feat_summary.loc[f, "raw_p95"]))
                      for f in candidates}
            allow_decrease = True

        result = greedy_optimize_snn2(model, st.session_state.slider_values, candidates, bounds,
                                       k=k, allow_decrease=allow_decrease)
        result["mode"] = opt_mode
        if opt_mode == "combined":
            subset_risk, applied = actionable_subset_risk_l2(
                model, st.session_state.slider_values, result["steps"], FEATURE_GROUPS_L2, ACTIONABLE_GROUPS_L2)
            result["actionable_subset_risk"] = subset_risk
            result["actionable_subset_steps"] = applied
        st.session_state.opt_result = result

    if st.session_state.get("opt_result"):
        result = st.session_state.opt_result
        arrow = {"up": "↑", "down": "↓"}
        if not result["steps"]:
            st.info("No available move reduces risk further from the current slider values.")
            return
        if result.get("mode") == "combined":
            m1, m2 = st.columns(2)
            m1.metric("Full optimized risk", f"{result['optimized_risk']:.1%}",
                       f"{result['optimized_risk'] - result['baseline_risk']:+.1%}")
            m2.metric("Actionable-only achievable", f"{result['actionable_subset_risk']:.1%}",
                       f"{result['actionable_subset_risk'] - result['baseline_risk']:+.1%} "
                       f"({len(result['actionable_subset_steps'])} real action(s))")
            steps_df = pd.DataFrame([
                {"Tag": "✅ Actionable" if is_actionable_step_l2(s["feature"], s["direction"], FEATURE_GROUPS_L2, ACTIONABLE_GROUPS_L2) else "⚠️ Correlational",
                 "Intervention": f"{arrow[s['direction']]} {FEATURE_LABELS_L2.get(s['feature'], s['feature'])}",
                 "New value": f"{s['to_value']:.1f}", "Risk after": f"{s['risk_after']:.1%}",
                 "Change": f"{s['risk_delta']:+.2%}"} for s in result["steps"]])
            st.dataframe(steps_df, hide_index=True, use_container_width=True)
            st.line_chart(pd.Series(result["risk_trajectory"], name="risk"))
            b1, b2 = st.columns(2)
            if b1.button("Apply all to sliders"):
                st.session_state.slider_values = dict(result["optimized_features"])
                st.session_state.opt_result = None
                st.rerun()
            if b2.button("Apply actionable-only to sliders"):
                updated = dict(st.session_state.slider_values)
                for s in result["actionable_subset_steps"]:
                    updated[s["feature"]] = s["to_value"]
                st.session_state.slider_values = updated
                st.session_state.opt_result = None
                st.rerun()
        else:
            st.metric("Optimized risk", f"{result['optimized_risk']:.1%}",
                       f"{result['optimized_risk'] - result['baseline_risk']:+.1%} over {len(result['steps'])} step(s)")
            steps_df = pd.DataFrame([
                {"Intervention": f"{arrow[s['direction']]} {FEATURE_LABELS_L2.get(s['feature'], s['feature'])}",
                 "New value": f"{s['to_value']:.1f}", "Risk after": f"{s['risk_after']:.1%}",
                 "Change": f"{s['risk_delta']:+.2%}"} for s in result["steps"]])
            st.dataframe(steps_df, hide_index=True, use_container_width=True)
            st.line_chart(pd.Series(result["risk_trajectory"], name="risk"))
            if result.get("mode") == "exploratory":
                st.caption("⚠️ May include decreases — see mode note above.")
            if st.button("Apply optimized values to sliders"):
                st.session_state.slider_values = dict(result["optimized_features"])
                st.session_state.opt_result = None
                st.rerun()


def build_delta_map(hex_df, results_df, basemap="light"):
    merged = hex_df.merge(results_df[["GRID_ID", "delta", "is_target"]], on="GRID_ID")
    merged["baseline_risk"] = merged["delta"]  # reuse the risk_col slot for delta

    max_abs = max(abs(merged["baseline_risk"].min()), abs(merged["baseline_risk"].max()), 1e-6)
    center_lat, center_lon = merged["centroid_lat"].mean(), merged["centroid_lon"].mean()
    tiles = "cartodbdark_matter" if basemap == "dark" else "cartodbpositron"
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=tiles)
    Fullscreen(position="topright").add_to(m)

    delta_cmap = cm.LinearColormap(colors=["#d73027", "#f7f7f7", "#1a9850"], vmin=-max_abs, vmax=max_abs)
    gj = hexes_to_geojson(merged, risk_col="baseline_risk")

    def style_fn(feature):
        risk = feature["properties"]["risk"]  # actually delta here
        is_target = merged.loc[merged["GRID_ID"] == feature["properties"]["GRID_ID"], "is_target"].iloc[0]
        return {
            "fillColor": delta_cmap(risk),
            "color": "#4d4d4d" if basemap == "light" else "#ffffff",
            "weight": 1.2 if is_target else 0.15,
            "fillOpacity": 0.85 if is_target else 0.35,
            "opacity": 0.6 if is_target else 0.2,
        }

    folium.GeoJson(
        gj, style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["GRID_ID", "risk"], aliases=["Hex", "ΔRisk (+ = improved)"],
            style=("background-color: white; color: #333; font-family: sans-serif; "
                   "font-size: 13px; padding: 6px; border-radius: 4px;"),
        ),
        name="delta",
    ).add_to(m)

    legend = delta_cmap.scale(-max_abs, max_abs)
    legend.caption = "ΔRisk from population optimization (bold outline = targeted hex)"
    legend = _legend_bottom_left(legend)
    legend.add_to(m)
    return m


def render_scenarios(model, hex_df, feat_summary, res, basemap):
    st.markdown(
        "Per the project's variable-role table: **Frozen context** (socioeconomic, "
        "demographic, crime, hazard, roadway, hydrology, land cover) is never changed. "
        "**Scenario inputs** = every other (Facilities/Services) variable is eligible — "
        "the doc names Affordable Housing / Food Access / selected service-amenity access "
        "as examples, not an exhaustive list, so all Facilities/Services variables are "
        "candidates here. Applied only to the highest-risk hexes, not the whole grid."
    )

    candidates = [f for f in model.base_features if FEATURE_GROUPS_L2.get(f) in ACTIONABLE_GROUPS_L2
                  and f in feat_summary.index]
    st.caption(f"Eligible scenario-input variables ({len(candidates)}): " +
               ", ".join(FEATURE_LABELS_L2.get(f, f) for f in candidates))

    c1, c2 = st.columns(2)
    top_pct = c1.slider("Apply to top X% highest-risk hexes", 5, 30, 10, key=f"scen_pct_{res}") / 100
    k = c2.slider("Max interventions per hex (K)", 1, 10, 3, key=f"scen_k_{res}")

    if st.button("Run population optimization", type="primary", key=f"scen_run_{res}"):
        bounds = {f: (0.0, float(feat_summary.loc[f, "raw_p95"])) for f in candidates}
        with st.spinner(f"Optimizing {int(top_pct * len(hex_df))} hexes..."):
            results, summary, usage = run_population_optimization(model, hex_df, top_pct, candidates, bounds, k=k)
        st.session_state.scenario_results = results
        st.session_state.scenario_summary = summary
        st.session_state.scenario_usage = usage

    if st.session_state.get("scenario_results") is not None:
        results = st.session_state.scenario_results
        summary = st.session_state.scenario_summary
        usage = st.session_state.scenario_usage

        st.markdown("**Population result (targeted hexes only)**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Hexes targeted", summary["n_hexes_targeted"])
        m2.metric("Mean baseline risk", f"{summary['mean_baseline_risk_targeted']:.1%}")
        m3.metric("Mean optimized risk", f"{summary['mean_optimized_risk_targeted']:.1%}",
                   f"{-summary['mean_reduction_targeted']:+.2%}")

        col_map, col_usage = st.columns([3, 2])
        with col_map:
            st.markdown("**ΔRisk map**")
            m = build_delta_map(hex_df, results, basemap=basemap)
            st_folium(m, height=480, width=None, returned_objects=[], key=f"scen_map_{res}")
        with col_usage:
            st.markdown("**Which variables got recommended, across targeted hexes**")
            if len(usage):
                st.bar_chart(usage.set_index("label")["pct_of_targeted_hexes"], height=480)
            else:
                st.info("No beneficial moves found for the targeted hexes.")


def main():
    st.title("LA Encampment Risk — Interactive Digital Twin (SNN-2)")

    with st.sidebar:
        st.header("Grid resolution")
        res = st.radio("Hexagon size", RESOLUTIONS, format_func=lambda r: RESOLUTION_LABELS[r], index=0)
        st.header("Map style")
        basemap = st.radio("Basemap", ["light", "dark"], format_func=str.title, horizontal=True)

    model = load_model(res)
    hex_df = load_hexdf(res)
    feat_summary = load_feature_summary(res)

    if "selected_grid_id" not in st.session_state or st.session_state.get("selected_res") != res:
        st.session_state.selected_res = res
        st.session_state.selected_grid_id = hex_df.loc[hex_df["baseline_risk"].idxmax(), "GRID_ID"]
        st.session_state.slider_values = None
        st.session_state.opt_result = None
        st.session_state.scenario_results = None
        st.session_state.scenario_summary = None
        st.session_state.scenario_usage = None

    with st.sidebar:
        st.header("Jump to a hex")
        grid_ids_sorted = sorted(hex_df["GRID_ID"].tolist())
        picked = st.selectbox("Search by hex ID", options=grid_ids_sorted,
                               index=grid_ids_sorted.index(st.session_state.selected_grid_id))
        if picked != st.session_state.selected_grid_id:
            st.session_state.selected_grid_id = picked
            st.session_state.slider_values = None
            st.session_state.opt_result = None

    selected_row = hex_df[hex_df["GRID_ID"] == st.session_state.selected_grid_id].iloc[0]
    if st.session_state.slider_values is None:
        st.session_state.slider_values = {
            f: float(selected_row[f]) if pd.notna(selected_row[f]) else 0.0 for f in model.base_features
        }

    risk, contribs = model.predict(st.session_state.slider_values)
    inter_contribs = {f: v for f, v in contribs.items() if f not in model.base_features}
    top_increase = max(inter_contribs.items(), key=lambda kv: kv[1], default=(None, 0.0))
    top_decrease = min(inter_contribs.items(), key=lambda kv: kv[1], default=(None, 0.0))

    # ---- Always-visible top section: map + headline numbers ----
    col_map, col_headline = st.columns([3, 2])

    with col_map:
        m = build_map(hex_df, st.session_state.selected_grid_id, current_risk_override=risk, basemap=basemap)
        map_state = st_folium(m, height=520, width=None, returned_objects=["last_active_drawing"])
        clicked = map_state.get("last_active_drawing")
        if clicked and clicked.get("properties", {}).get("GRID_ID"):
            clicked_gid = clicked["properties"]["GRID_ID"]
            if clicked_gid != st.session_state.selected_grid_id:
                st.session_state.selected_grid_id = clicked_gid
                st.session_state.slider_values = None
                st.session_state.opt_result = None
                st.rerun()

    with col_headline:
        gid = st.session_state.selected_grid_id
        centroid_note = ("actual (averaged detection points)" if selected_row["centroid_source"] == "detection_avg"
                          else "reconstructed (no detections in this hex)")
        st.subheader(f"Hex {gid}")
        st.caption(f"Centroid: {centroid_note}")

        baseline_risk = float(selected_row["baseline_risk"])
        delta = risk - baseline_risk
        m1, m2 = st.columns(2)
        m1.metric("Predicted risk", f"{risk:.1%}", f"{delta:+.1%} vs. real values")
        m2.metric("Baseline (real values)", f"{baseline_risk:.1%}")

        if top_increase[0] and top_increase[1] > 0:
            a, b = top_increase[0].split("__x__")
            st.error(
                f"**Top interaction increasing risk:**\n\n"
                f"{interaction_label(a, b)}\n\n"
                f"raising risk by {top_increase[1]:.1%} (contribution)"
            )
        if top_decrease[0] and top_decrease[1] < 0:
            a, b = top_decrease[0].split("__x__")
            st.success(
                f"**Top interaction lowering risk:**\n\n"
                f"{interaction_label(a, b)}\n\n"
                f"lowering risk by {abs(top_decrease[1]):.1%} (contribution)"
            )

        if st.button("Reset sliders to real values", use_container_width=True):
            st.session_state.slider_values = None
            st.session_state.opt_result = None
            st.rerun()

    st.divider()

    # ---- Everything else in tabs, so a presenter clicks instead of scrolls ----
    tab_scenarios, tab_sliders, tab_contrib, tab_opt = st.tabs(
        ["🏘️ Scenarios", "🎚️ Adjust Variables", "📊 Why This Score", "🎯 Optimize This Hex"]
    )
    with tab_scenarios:
        render_scenarios(model, hex_df, feat_summary, res, basemap)
    with tab_sliders:
        render_sliders(model, feat_summary, res, gid)
    with tab_contrib:
        render_contributions(model, contribs)
    with tab_opt:
        render_optimizer(model, feat_summary, res, gid)


if __name__ == "__main__":
    main()
