"""
Streamlit app for the Privacy-Preserving Healthcare AI Benchmark.

Upload any tabular CSV files (one per hospital/site, or a single file to be split into
simulated sites), pick the outcome column, and the app trains and benchmarks:
centralized, FedAvg, SecAgg, DP, DP+SecAgg, plus membership inference, model inversion
and gradient-leakage attacks.

Run:
    pip install streamlit pandas numpy altair
    streamlit run streamlit_app.py

Keep this file next to fel_privacy_benchmark.py and your V19 script (fel_v19.py).
"""
import json
import os
import re
import subprocess
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = ["DP", "DP+SecAgg"]
TARGET_HINTS = ["target", "label", "y", "class", "outcome", "num", "condition", "disease",
                "diagnosis", "heartdisease", "heart_disease", "cardio"]

st.set_page_config(page_title="Privacy-Preserving Healthcare AI Benchmark", layout="wide")
st.title("Privacy-Preserving Healthcare AI Benchmark")
st.caption("Upload CSV files, then compare Centralized, FedAvg, Differential Privacy and Secure "
           "Aggregation, with simulated membership inference, model inversion and gradient leakage attacks.")


# ------------------------------------------------------------------ helpers
def abs_dir(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


@st.cache_data(show_spinner=False)
def read_csv_cached(path, mtime):
    return pd.read_csv(path)


def read_preview(uploaded, nrows=300):
    uploaded.seek(0)
    try:
        df = pd.read_csv(uploaded, sep=None, engine="python", nrows=nrows,
                         na_values=["?", "NA", "N/A", "", "null", "NULL"])
    finally:
        uploaded.seek(0)
    return df


def safe_name(name, used):
    base = re.sub(r"[^A-Za-z0-9_]+", "_", os.path.splitext(name)[0]).strip("_") or "site"
    n, i = base, 2
    while n in used:
        n, i = f"{base}_{i}", i + 1
    used.add(n)
    return n


def guess_target(cols):
    low = {str(c).strip().lower().replace(" ", "_"): c for c in cols}
    for h in TARGET_HINTS:
        if h in low:
            return low[h]
    return cols[-1]


METRIC_LABELS = {
    "f1": "F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "auc": "ROC-AUC",
    "mia_adv": "Membership inference advantage", "inv_adv": "Model inversion advantage over baseline",
    "leak_cos": "Gradient leakage (cosine similarity)", "leak_success": "Gradient leakage success rate",
}


def sweep_chart(df, col, title):
    d = df[df.family.isin(SWEEP)].copy()
    d = d[np.isfinite(d["epsilon"])]
    d["lo"] = d[col + "_mean"] - d[col + "_std"]
    d["hi"] = d[col + "_mean"] + d[col + "_std"]
    x = alt.X("epsilon:Q", scale=alt.Scale(type="log"), title="privacy budget ε (lower = more private)")
    base = alt.Chart(d).encode(x=x, color=alt.Color("family:N", title="Sweep"))
    line = base.mark_line(point=True).encode(
        y=alt.Y(col + "_mean:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=["config", "epsilon", alt.Tooltip(col + "_mean:Q", format=".3f")])
    band = base.mark_errorbar().encode(y=alt.Y("lo:Q", title=title), y2="hi:Q")
    layers = [band, line]
    b = df[~df.family.isin(SWEEP)]
    if len(b):
        layers.append(alt.Chart(b).mark_rule(strokeDash=[5, 4]).encode(
            y=alt.Y(col + "_mean:Q"), color=alt.Color("config:N", title="Baseline"),
            tooltip=["config", alt.Tooltip(col + "_mean:Q", format=".3f")]))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=380)


# ------------------------------------------------------------------ sidebar: results location
active = st.session_state.get("active_dir")
st.sidebar.header("Results")
out_dir = st.sidebar.text_input("Results folder", value=active or "benchmark_outputs")
up_sum = st.sidebar.file_uploader("...or upload results_summary.csv", type="csv")
up_raw = st.sidebar.file_uploader("...and results_raw.csv (optional)", type="csv")


def load_results():
    if up_sum is not None:
        return pd.read_csv(up_sum), (pd.read_csv(up_raw) if up_raw is not None else None), None
    sp = os.path.join(abs_dir(out_dir), "results_summary.csv")
    rp = os.path.join(abs_dir(out_dir), "results_raw.csv")
    ip = os.path.join(abs_dir(out_dir), "dataset_info.json")
    if not os.path.exists(sp):
        return None, None, None
    info = json.load(open(ip)) if os.path.exists(ip) else None
    return (read_csv_cached(sp, os.path.getmtime(sp)),
            read_csv_cached(rp, os.path.getmtime(rp)) if os.path.exists(rp) else None, info)


summary, raw, info = load_results()

tabs = st.tabs(["1. Upload & run", "Frontier", "Attacks", "Cost", "Budget explorer", "Data"])

# ======================================================================= TAB 1: UPLOAD & RUN
with tabs[0]:
    st.subheader("Upload your data")
    st.write("Each CSV is treated as one site (hospital). Files must describe the same kind of patients "
             "with overlapping column names and a binary (or low-cardinality) outcome column. "
             "With **one** file, the app splits it into simulated sites.")
    files = st.file_uploader("CSV files", type=["csv", "txt", "data"], accept_multiple_files=True)

    previews, cols_union = {}, []
    if files:
        for f in files:
            try:
                previews[f.name] = read_preview(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
        for name, df in previews.items():
            with st.expander(f"{name}: preview ({df.shape[1]} columns)"):
                st.dataframe(df.head(10), use_container_width=True)
            for c in df.columns:
                if c not in cols_union:
                    cols_union.append(c)

    if previews:
        st.subheader("Settings")
        c1, c2, c3 = st.columns(3)
        first_cols = list(next(iter(previews.values())).columns)
        options = ["(auto-detect)"] + [str(c) for c in cols_union]
        default = options.index(str(guess_target(first_cols))) if str(guess_target(first_cols)) in options else 0
        target = c1.selectbox("Outcome / target column", options, index=default,
                              help="Binary preferred. Multi-level targets are binarised (lowest level = 0).")
        simulate_k = 0
        if len(previews) == 1:
            simulate_k = c2.number_input("Split into K simulated sites", 2, 10, 4)
        max_rows = c3.number_input("Max rows per site (0 = all)", 0, 1_000_000, 5000, step=500,
                                   help="Subsample for speed. Training is NumPy-only and slows with size.")
        c4, c5, c6 = st.columns(3)
        quick = c4.checkbox("Quick mode (1 seed, 10 rounds, 3 ε)", value=True)
        generic = c5.checkbox("Treat as generic tabular data", value=True,
                              help="Off = try V19's strict heart-disease schema first, fall back to generic.")
        module = c6.text_input("V19 module name", "fel_v19")
        c7, c8, c9 = st.columns(3)
        seeds = c7.number_input("Seeds", 1, 10, 3, disabled=quick)
        rounds = c8.number_input("Rounds", 5, 200, 30, disabled=quick)
        clip = c9.number_input("DP clip norm", 0.1, 20.0, 5.0)
        eps_txt = st.text_input("ε values (space separated)", "1 5 10 50 200 1000", disabled=quick)
        sens = st.text_input("Hidden attribute for model inversion ('auto' picks a binary feature)", "auto")

        if st.button("Run benchmark on these files", type="primary"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            up_dir = os.path.join(HERE, "uploads", stamp)
            os.makedirs(up_dir, exist_ok=True)
            used, data_args = set(), []
            for f in files:
                n = safe_name(f.name, used)
                p = os.path.join(up_dir, n + ".csv")
                with open(p, "wb") as fh:
                    fh.write(f.getbuffer())
                data_args.append(f"{n}={p}")
            run_rel = os.path.join("runs", stamp)
            cmd = [sys.executable, "-u", os.path.join(HERE, "fel_privacy_benchmark.py"),
                   "--module", module, "--data", *data_args, "--dp_clip", str(clip),
                   "--sensitive_feature", sens, "--max_rows", str(int(max_rows)),
                   "--out", os.path.join(HERE, run_rel)]
            if target != "(auto-detect)":
                cmd += ["--target", target]
            if generic:
                cmd.append("--generic")
            if simulate_k:
                cmd += ["--simulate_clients", str(int(simulate_k))]
            if quick:
                cmd.append("--quick")
            else:
                cmd += ["--seeds", str(int(seeds)), "--rounds", str(int(rounds)), "--eps", *eps_txt.split()]

            box, lines, t0 = st.empty(), [], time.time()
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with st.spinner("Training and attacking... this can take a while"):
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    box.code("\n".join(lines[-40:]))
            proc.wait()
            st.session_state["last_log"] = "\n".join(lines)
            if proc.returncode == 0:
                st.session_state["active_dir"] = run_rel
                st.session_state["last_status"] = f"Finished in {time.time() - t0:.0f}s."
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Benchmark failed. Read the last lines of the log below.")
                st.code("\n".join(lines[-25:]))
    else:
        st.info("Upload at least one CSV to begin.")

    if st.session_state.get("last_status"):
        st.success(st.session_state["last_status"] + " Results are in the other tabs.")
    if st.session_state.get("last_log"):
        with st.expander("Last run log"):
            st.code(st.session_state["last_log"][-6000:])

# ======================================================================= results tabs
if summary is None:
    for i in range(1, 6):
        with tabs[i]:
            st.info("No results yet. Upload CSV files in the first tab and run the benchmark, "
                    "or load a results folder / summary CSV from the sidebar.")
    st.stop()

summary["epsilon"] = pd.to_numeric(summary["epsilon"], errors="coerce").fillna(np.inf)

with tabs[1]:
    m = st.selectbox("Utility metric", ["f1", "accuracy", "balanced_accuracy", "auc"], format_func=METRIC_LABELS.get)
    st.altair_chart(sweep_chart(summary, m, METRIC_LABELS[m]), use_container_width=True)
    st.caption("Lines: DP and DP+SecAgg sweeps (band = ±1 std over seeds). Dashed rules: configurations with no "
               "finite ε. DP and DP+SecAgg have near-identical utility because SecAgg masks cancel exactly.")
    if "on_frontier" in summary:
        f = summary[summary["on_frontier"] == True]
        st.markdown("**Pareto-optimal configurations** (no config has lower-or-equal ε and higher F1)")
        st.dataframe(f[["config", "family", "epsilon", "f1_mean", "accuracy_mean", "auc_mean"]].round(4),
                     use_container_width=True)

with tabs[2]:
    a = st.selectbox("Attack metric", ["mia_adv", "inv_adv", "leak_cos", "leak_success"], format_func=METRIC_LABELS.get)
    st.altair_chart(sweep_chart(summary, a, METRIC_LABELS[a]), use_container_width=True)
    with st.expander("How to read this"):
        st.markdown(
            "- **Membership inference**: loss-threshold attack, train rows vs test rows. 0 = no leakage; "
            "values near 0 are common for small tabular models.\n"
            "- **Model inversion**: guesses a hidden binary feature from the model likelihood and the label; "
            "advantage is measured against a majority-class baseline.\n"
            "- **Gradient leakage**: an honest-but-curious server reconstructs a patient's features from the "
            "update it sees. With plain FedAvg and DP-only it sees individual updates, so leakage stays high; "
            "SecAgg hides them, and DP+SecAgg also adds noise.\n"
            "- Centralized is scored as full exposure because raw rows are uploaded.")
    st.dataframe(summary[["config", "family", "epsilon", "mia_adv_mean", "inv_adv_mean", "leak_cos_mean",
                          "leak_success_mean"]].round(4), use_container_width=True)

with tabs[3]:
    d = summary.reset_index(drop=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Communication (MB, log scale)**")
    c1.altair_chart(alt.Chart(d).mark_bar().encode(
        x=alt.X("config:N", sort=None, title=None), y=alt.Y("comm_mb_mean:Q", scale=alt.Scale(type="log")),
        color="family:N", tooltip=["config", alt.Tooltip("comm_mb_mean:Q", format=".2f")]).properties(height=360),
        use_container_width=True)
    c2.markdown("**Training time (s)**")
    c2.altair_chart(alt.Chart(d).mark_bar().encode(
        x=alt.X("config:N", sort=None, title=None), y="train_time_s_mean:Q",
        color="family:N", tooltip=["config", alt.Tooltip("train_time_s_mean:Q", format=".1f")]).properties(height=360),
        use_container_width=True)
    st.caption("Communication is modeled: float64 parameters up/down per round; SecAgg uploads one share per peer. "
               "Centralized counts the one-time raw-data upload.")

with tabs[4]:
    st.subheader("Which configuration fits my requirements?")
    fin = summary[np.isfinite(summary["epsilon"])]
    lo = float(fin["epsilon"].min()) if len(fin) else 0.1
    hi = float(fin["epsilon"].max()) if len(fin) else 1000.0
    max_eps = st.slider("Maximum acceptable ε", lo, hi, hi) if hi > lo else hi
    need_hide = st.checkbox("Require that individual updates are hidden from the server (SecAgg)")
    max_leak = st.slider("Maximum gradient-leakage cosine similarity", 0.0, 1.0, 1.0, 0.05)
    cand = summary[(summary["epsilon"] <= max_eps) & (summary["leak_cos_mean"] <= max_leak)]
    if need_hide:
        cand = cand[cand["family"].isin(["SecAgg", "DP+SecAgg"])]
    cand = cand.sort_values("f1_mean", ascending=False)
    if cand.empty:
        st.warning("No configuration satisfies these constraints. Relax the budget or the leakage cap.")
    else:
        best = cand.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best configuration", str(best["config"]))
        k2.metric("F1", f"{best['f1_mean']:.3f}")
        k3.metric("ε", "∞" if not np.isfinite(best["epsilon"]) else f"{best['epsilon']:g}")
        k4.metric("Leakage (cos)", f"{best['leak_cos_mean']:.2f}")
        st.dataframe(cand[["config", "family", "epsilon", "f1_mean", "leak_cos_mean", "mia_adv_mean",
                           "comm_mb_mean", "train_time_s_mean"]].round(4), use_container_width=True)

with tabs[5]:
    if info:
        st.markdown(f"**Dataset** ({info['mode']} loader, hidden attribute for inversion: "
                    f"`{info.get('sensitive_feature') or 'none'}`)")
        st.dataframe(pd.DataFrame(info["sites"]), use_container_width=True)
        st.caption(f"{len(info['features'])} features: " + ", ".join(info["features"]))
    st.markdown("**Summary**")
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download summary CSV", summary.to_csv(index=False), "results_summary.csv")
    if raw is not None:
        st.markdown("**Per-seed results**")
        st.dataframe(raw, use_container_width=True)
        st.download_button("Download raw CSV", raw.to_csv(index=False), "results_raw.csv")
