#!/usr/bin/env python3
"""
PRIVACY-PRESERVING HEALTHCARE AI BENCHMARK
==========================================
Built on top of FEL V19 (save your V19 script as `fel_v19.py` next to this file,
or pass --module <name>). V19 is imported, not modified.

Configurations benchmarked
  1. Centralized ML              (pooled data; raw rows leave the hospitals)
  2. FedAvg                      (no privacy mechanism)
  3. FedAvg + SecAgg             (secure aggregation only)
  4. FedAvg + DP(eps)            (formal client-level DP, epsilon sweep)
  5. FedAvg + DP(eps) + SecAgg   (combined, epsilon sweep)

Metrics per configuration (mean/std over seeds)
  accuracy, balanced accuracy, F1, AUC, privacy budget (epsilon),
  training time, communication overhead (MB)

Simulated attacks (run against every trained configuration)
  - Membership inference : loss-threshold attack on the released global model
  - Model inversion      : attribute inference (Fredrikson-style) on a sensitive feature
  - Gradient leakage     : honest-but-curious server reconstructs a patient's
                           feature vector from the update it observes

Outputs (in --out): results_raw.csv, results_summary.csv, benchmark_report.json,
                    privacy_utility_frontier.png

Usage:
    python fel_privacy_benchmark.py --quick
    python fel_privacy_benchmark.py --data siteA.csv siteB.csv siteC.csv --generic --target outcome
    python fel_privacy_benchmark.py --data one_big_file.csv --simulate_clients 4 --max_rows 5000
    python fel_privacy_benchmark.py --seeds 3 --rounds 30 --eps 1 5 10 50 200 1000
"""
import argparse
import contextlib
import copy
import importlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

fel = None  # the imported V19 module (set in main)
INF = float("inf")


# ============================================================
# HELPERS
# ============================================================

@contextlib.contextmanager
def override_config(**kw):
    """Temporarily change V19's CONFIG entries."""
    saved = {k: copy.deepcopy(fel.CONFIG[k]) for k in kw}
    fel.CONFIG.update(kw)
    try:
        yield
    finally:
        fel.CONFIG.update(saved)


def build_models(lin_s, dnn_s, meta_s, d):
    C = fel.CONFIG
    lin = fel.FedLinear(d, C["linear_lr"]); lin.set_global(lin_s)
    dnn = fel.FedDNN(2 * d, C["dnn_hidden1"], C["dnn_hidden2"], C["dnn_lr"]); dnn.set_global(dnn_s)
    meta = fel.FedMeta(C["meta_lr"]); meta.set_global(meta_s)
    return lin, dnn, meta


def split_arrays(c, split):
    return (getattr(c, f"X_{split}_linear"),
            getattr(c, f"X_{split}_dnn"),
            getattr(c, f"y_{split}"))


def split_predictions(models, clients, split):
    """Pooled labels, stacking prob, and base-average prob over all clients."""
    lin, dnn, meta = models
    ys, pl, pd_ = [], [], []
    for c in clients:
        Xl, Xd, y = split_arrays(c, split)
        pl.append(lin.predict_p(Xl)); pd_.append(dnn.predict_p(Xd)); ys.append(y)
    y = np.concatenate(ys); pl = np.concatenate(pl); pd_ = np.concatenate(pd_)
    ps = meta.predict_p(np.column_stack([pl, pd_]))
    return y, ps, 0.5 * (pl + pd_)


def select_on_validation(models, clients):
    y, ps, pb = split_predictions(models, clients, "val")
    score, mix, thr, _, _, _ = fel.choose_blend(y, ps, pb)
    return score, mix, thr


class GlobalModel:
    """Final released model: states + blend mix + decision threshold."""

    def __init__(self, lin_s, dnn_s, meta_s, mix, thr, d):
        self.lin_state, self.dnn_state, self.meta_state = lin_s, dnn_s, meta_s
        self.mix, self.thr, self.d = float(mix), float(thr), d
        self.models = build_models(lin_s, dnn_s, meta_s, d)

    def proba(self, Xl, Xd):
        lin, dnn, meta = self.models
        pl, pd_ = lin.predict_p(Xl), dnn.predict_p(Xd)
        ps = meta.predict_p(np.column_stack([pl, pd_]))
        return self.mix * ps + (1.0 - self.mix) * 0.5 * (pl + pd_)

    def test_metrics(self, clients):
        y, ps, pb = split_predictions(self.models, clients, "test")
        p = self.mix * ps + (1.0 - self.mix) * pb
        return fel.metric_dict(y, p, self.thr)


def n_params(state_blocks):
    return int(sum(np.asarray(state_blocks[k]).size for k in state_blocks))


# ============================================================
# TRAINING: FEDERATED CONFIGURATIONS
# ============================================================

def run_federated(clients, d, rounds, dp, epsilon, secagg, dp_clip):
    """Fixed number of rounds (so the DP accountant is exact), best-validation checkpoint."""
    n_train = np.asarray([c.meta_info.n_train for c in clients], dtype=float)
    max_w = float((n_train / n_train.sum()).max())

    with override_config(num_rounds=rounds,
                         privacy_mode="formal_dp" if dp else "utility",
                         target_epsilon=float(epsilon) if dp else fel.CONFIG["target_epsilon"],
                         use_secure_aggregation=bool(secagg),
                         dp_clip_norm=float(dp_clip)):
        server = fel.FELServer(d, "fedavg")
        server.aggregator.num_clients = len(clients)
        server.configure_privacy(max_w)

        best, train_time, eps_final = None, 0.0, INF
        for _ in range(rounds):
            t0 = time.perf_counter()
            gs = server.state()
            updates = [c.local_round(gs, "fedavg") for c in clients]
            eps_r, _, _ = server.aggregate(updates)
            train_time += time.perf_counter() - t0
            if dp:
                eps_final = eps_r

            models = build_models(server.global_linear, server.global_dnn, server.global_meta, d)
            score, mix, thr = select_on_validation(models, clients)
            if best is None or score > best[0] + 1e-5:
                best = (score, mix, thr, server.state())

        _, mix, thr, st = best
        gm = GlobalModel(st["linear"], st["dnn"], st["meta"], mix, thr, d)

        # communication model (float64 parameters)
        up_bytes = 8 * (n_params({k: server.global_linear[k] for k in ["w", "b"]}) +
                        n_params({k: server.global_dnn[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}) +
                        n_params({k: server.global_meta[k] for k in ["w", "b"]}))
        k = len(clients)
        # additive secret sharing: each client sends one share to every peer + one to the server
        per_client_up = up_bytes * (k if secagg else 1)
        comm_mb = rounds * k * (per_client_up + up_bytes) / 1e6

        return {"gm": gm, "train_time": train_time, "epsilon": eps_final,
                "sigma": float(server.sigma) if dp else 0.0,
                "sens": float(server.accountant.sensitivity) if dp else 0.0,
                "comm_mb": comm_mb, "rounds": rounds}


# ============================================================
# TRAINING: CENTRALIZED BASELINE
# ============================================================

def run_centralized(clients, d, rounds):
    C = fel.CONFIG
    Xl = np.vstack([c.X_train_linear for c in clients])
    Xd = np.vstack([c.X_train_dnn for c in clients])
    y = np.concatenate([c.y_train for c in clients])
    cw = fel.class_weights(y, C["class_weight_cap"]) if C["use_class_balancing"] else {0: 1.0, 1: 1.0}

    lin = fel.FedLinear(d, C["linear_lr"])
    dnn = fel.FedDNN(2 * d, C["dnn_hidden1"], C["dnn_hidden2"], C["dnn_lr"])
    meta = fel.FedMeta(C["meta_lr"])
    st = {"linear": lin.state(False), "dnn": dnn.state(False), "meta": meta.state(False)}

    best, train_time = None, 0.0
    for _ in range(rounds):
        t0 = time.perf_counter()
        lin_u = lin.local_update(Xl, y, st["linear"], "fedavg", C["linear_epochs"], C["fedprox_mu"], cw)
        dnn_u = dnn.local_update(Xd, y, st["dnn"], "fedavg", C["dnn_epochs"], C["fedprox_mu"], cw)
        bl, bd, _ = build_models(st["linear"], st["dnn"], st["meta"], d)
        P = np.column_stack([bl.predict_p(Xl), bd.predict_p(Xd)])
        meta_u = meta.local_update(P, y, st["meta"], "fedavg", C["meta_epochs"], C["fedprox_mu"], cw)
        st = {"linear": lin_u, "dnn": dnn_u, "meta": meta_u}
        train_time += time.perf_counter() - t0

        models = build_models(st["linear"], st["dnn"], st["meta"], d)
        score, mix, thr = select_on_validation(models, clients)
        if best is None or score > best[0] + 1e-5:
            best = (score, mix, thr, copy.deepcopy(st))

    _, mix, thr, s = best
    gm = GlobalModel(s["linear"], s["dnn"], s["meta"], mix, thr, d)
    comm_mb = sum(c.meta_info.n_train * (d + 1) * 8 for c in clients) / 1e6  # raw rows uploaded once
    return {"gm": gm, "train_time": train_time, "epsilon": INF, "sigma": 0.0,
            "sens": 0.0, "comm_mb": comm_mb, "rounds": rounds}


# ============================================================
# ATTACK 1: MEMBERSHIP INFERENCE (loss-threshold)
# ============================================================

def membership_inference(gm, clients, rng):
    def losses(split):
        out = []
        for c in clients:
            Xl, Xd, y = split_arrays(c, split)
            p = np.clip(gm.proba(Xl, Xd), 1e-7, 1 - 1e-7)
            out.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return np.concatenate(out)

    lm, ln = losses("train"), losses("test")           # members vs non-members
    n = min(len(lm), len(ln))
    if n < 10:
        return {"mia_auc": np.nan, "mia_adv": np.nan}
    lm = rng.choice(lm, n, replace=False)
    ln = rng.choice(ln, n, replace=False)
    labels = np.r_[np.ones(n), np.zeros(n)]
    score = -np.r_[lm, ln]                              # lower loss => more likely member
    fpr, tpr, _ = roc_curve(labels, score)
    return {"mia_auc": float(roc_auc_score(labels, score)),
            "mia_adv": float(np.max(tpr - fpr))}


# ============================================================
# ATTACK 2: MODEL INVERSION (attribute inference)
# ============================================================

def attribute_inference(gm, clients, features, target_feature="sex"):
    """Given all other features + the true label, pick the value of a hidden binary
    feature that maximises model likelihood (with the population prior)."""
    if target_feature not in features:
        return {"inv_acc": np.nan, "inv_baseline": np.nan, "inv_adv": np.nan}
    j = features.index(target_feature)
    correct = base_correct = total = 0
    for c in clients:
        if target_feature not in c.X_train_raw.columns:
            continue
        col = pd.to_numeric(c.X_train_raw[target_feature], errors="coerce")
        vals = sorted(col.dropna().unique())
        if len(vals) != 2:
            continue
        valid = ~col.isna().values
        true_idx = (col.values == vals[1]).astype(int)
        mu = c.preprocessor["mean"][target_feature]
        sd = c.preprocessor["std"][target_feature]
        prior = np.array([np.mean(true_idx[valid] == 0), np.mean(true_idx[valid] == 1)])
        prior = np.clip(prior, 1e-6, 1.0)
        y = c.y_train
        scores = []
        for k, v in enumerate(vals):
            cv = (v - mu) / sd
            Xl, Xd = c.X_train_linear.copy(), c.X_train_dnn.copy()
            Xl[:, j] = cv
            Xd[:, j] = cv
            p = np.clip(gm.proba(Xl, Xd), 1e-7, 1 - 1e-7)
            scores.append(np.log(np.where(y == 1, p, 1 - p)) + np.log(prior[k]))
        pred = np.argmax(np.vstack(scores), axis=0)
        maj = int(prior[1] >= prior[0])
        correct += int(np.sum((pred == true_idx)[valid]))
        base_correct += int(np.sum((true_idx == maj)[valid]))
        total += int(valid.sum())
    if total == 0:
        return {"inv_acc": np.nan, "inv_baseline": np.nan, "inv_adv": np.nan}
    acc, base = correct / total, base_correct / total
    return {"inv_acc": acc, "inv_baseline": base, "inv_adv": acc - base}


# ============================================================
# ATTACK 3: GRADIENT LEAKAGE (honest-but-curious server)
# ============================================================
# Single-sample update of the logistic model: delta = -lr * (p - y) * [x, 1].
# Hence x = delta_w / delta_b exactly when the server sees ONE client's update.
#   no SecAgg          : server sees the individual (clipped) client update
#   SecAgg             : server only sees the weighted sum over all clients
#   DP (central)       : calibrated noise is added to the aggregate before release,
#                        so it only affects what the server sees if SecAgg hides
#                        the individual updates (DP-only trusts the server).

def gradient_leakage(gm, clients, d, secagg, dp, sigma, sens, clip, rng, n_targets=200):
    lr = fel.CONFIG["linear_lr"]
    lin = fel.FedLinear(d, lr); lin.set_global(gm.lin_state)
    n_train = np.asarray([c.meta_info.n_train for c in clients], dtype=float)
    w = n_train / n_train.sum()

    def delta(x, y):
        p = lin.predict_p(x[None, :])[0]
        v = -lr * (p - y) * np.concatenate([x, [1.0]])
        return fel.clip_vec(v, clip) if dp else v

    cos_l, rel_l = [], []
    for _ in range(n_targets):
        ci = int(rng.integers(len(clients)))
        k = int(rng.integers(len(clients[ci].y_train)))
        x = clients[ci].X_train_linear[k]
        obs = delta(x, clients[ci].y_train[k])
        if secagg:
            obs = w[ci] * obs
            for cj, o in enumerate(clients):
                if cj == ci:
                    continue
                kk = int(rng.integers(len(o.y_train)))
                obs = obs + w[cj] * delta(o.X_train_linear[kk], o.y_train[kk])
            if dp:
                obs = obs + rng.normal(0.0, sigma * sens, obs.shape)
        x_hat = np.zeros(d) if abs(obs[-1]) < 1e-12 else np.clip(obs[:-1] / obs[-1], -10, 10)
        nx, nh = np.linalg.norm(x), np.linalg.norm(x_hat)
        cos_l.append(float(x @ x_hat / (nx * nh)) if nx > 1e-9 and nh > 1e-9 else 0.0)
        rel_l.append(float(np.linalg.norm(x_hat - x) / (nx + 1e-12)))
    cos_l, rel_l = np.asarray(cos_l), np.asarray(rel_l)
    return {"leak_cos": float(cos_l.mean()),
            "leak_relerr": float(np.median(rel_l)),
            "leak_success": float(np.mean(cos_l > 0.95))}


# ============================================================
# CONFIGURATIONS, FRONTIER, REPORTING
# ============================================================

def build_configs(eps_list):
    cfgs = [
        dict(name="Centralized", family="Centralized", kind="central", dp=False, secagg=False, eps=INF),
        dict(name="FedAvg", family="FedAvg", kind="fed", dp=False, secagg=False, eps=INF),
        dict(name="FedAvg+SecAgg", family="SecAgg", kind="fed", dp=False, secagg=True, eps=INF),
    ]
    for e in eps_list:
        cfgs.append(dict(name=f"FedAvg+DP(eps={e:g})", family="DP", kind="fed", dp=True, secagg=False, eps=float(e)))
    for e in eps_list:
        cfgs.append(dict(name=f"FedAvg+DP+SecAgg(eps={e:g})", family="DP+SecAgg", kind="fed", dp=True, secagg=True, eps=float(e)))
    return cfgs


def pareto_flags(eps, util):
    """True where no other point has lower-or-equal epsilon AND strictly higher utility."""
    e = np.where(np.isfinite(eps), eps, 1e12)
    order = np.argsort(e, kind="stable")
    flags = np.zeros(len(e), dtype=bool)
    best = -INF
    for i in order:
        if util[i] > best + 1e-9:
            flags[i] = True
            best = util[i]
    return flags


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "train_time_s", "comm_mb",
           "mia_auc", "mia_adv", "inv_acc", "inv_baseline", "inv_adv",
           "leak_cos", "leak_relerr", "leak_success"]


def summarize(raw):
    g = raw.groupby("config", sort=False)
    mean = g[METRICS].mean().add_suffix("_mean")
    std = g[METRICS].std(ddof=0).add_suffix("_std")
    meta = g[["family", "epsilon"]].first()
    s = meta.join(mean).join(std)
    s["on_frontier"] = pareto_flags(s["epsilon"].values, s["f1_mean"].values)
    return s


def make_plots(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    styles = {"DP": "o-", "DP+SecAgg": "s--"}
    base = summary[summary.family.isin(["Centralized", "FedAvg", "SecAgg"])]

    def panel(a, col, title, ylabel):
        for fam, st in styles.items():
            s = summary[summary.family == fam].sort_values("epsilon")
            if len(s):
                a.errorbar(s.epsilon, s[col + "_mean"], yerr=s[col + "_std"], fmt=st, capsize=3, label=fam)
        for name, r in base.iterrows():
            a.axhline(r[col + "_mean"], ls=":", lw=1.2, color={"Centralized": "k", "FedAvg": "tab:green", "SecAgg": "tab:red"}[r.family], label=name)
        a.set_xscale("log"); a.set_xlabel("privacy budget epsilon (lower = more private)")
        a.set_ylabel(ylabel); a.set_title(title); a.grid(alpha=0.3)

    panel(ax[0, 0], "f1", "Privacy-utility frontier (F1)", "test F1")
    ax[0, 0].legend(fontsize=7)
    panel(ax[0, 1], "mia_adv", "Membership inference advantage", "advantage (0 = no leakage)")
    panel(ax[1, 0], "leak_cos", "Gradient leakage (cosine sim. of reconstruction)", "cosine similarity")

    fam = summary.groupby("family", sort=False)[["comm_mb_mean", "train_time_s_mean"]].mean()
    x = np.arange(len(fam))
    ax[1, 1].bar(x - 0.2, fam.comm_mb_mean, 0.4, label="communication (MB)")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(fam.index, rotation=20)
    ax[1, 1].set_ylabel("MB"); ax[1, 1].set_title("Cost: communication and training time")
    twin = ax[1, 1].twinx()
    twin.bar(x + 0.2, fam.train_time_s_mean, 0.4, color="tab:orange", label="train time (s)")
    twin.set_ylabel("seconds")
    fig.tight_layout()
    path = os.path.join(out_dir, "privacy_utility_frontier.png")
    fig.savefig(path, dpi=160)
    print("  plot    :", path)



# ============================================================
# DATA LOADING: strict V19 clinical schema OR generic any-CSV
# ============================================================

_ALIAS = None


def alias_lookup():
    global _ALIAS
    if _ALIAS is None:
        _ALIAS = {}
        for canon, aliases in fel.FEATURE_ALIASES.items():
            for a in aliases:
                _ALIAS[fel.normalize_feature_name(a)] = canon
    return _ALIAS


def load_generic_csv(path, name, target=None):
    """Load an arbitrary tabular CSV with a (near-)binary outcome column."""
    df = pd.read_csv(path, sep=None, engine="python",
                     na_values=["?", "NA", "N/A", "", "null", "NULL"])
    df.columns = [fel.normalize_feature_name(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    if df.shape[0] < 40 or df.shape[1] < 3:
        raise ValueError(f"{name}: needs at least 40 rows and 3 columns (got {df.shape[0]}x{df.shape[1]}).")

    tcol = None
    if target:
        t = fel.normalize_feature_name(target)
        if t in df.columns:
            tcol = t
    if tcol is None:
        tcol = fel.infer_target_column(df)
    if tcol is None:
        tcol = df.columns[-1]
    df = df[df[tcol].notna()].reset_index(drop=True)

    try:
        y, mapping = fel.normalize_target(df[tcol], tcol)
    except ValueError:
        v = df[tcol]
        nums = pd.to_numeric(v, errors="coerce")
        base = nums if nums.notna().all() else pd.Series(pd.factorize(v.astype(str))[0], index=v.index)
        if base.nunique() > 10:
            raise ValueError(f"{name}: target '{tcol}' has {base.nunique()} levels; pick a binary or "
                             f"low-cardinality target column.")
        y = (base.values != base.min()).astype(int)
        mapping = f"multi-level: lowest level->0, others->1"
    y = np.asarray(y, dtype=int)
    if min(np.sum(y == 0), np.sum(y == 1)) < 10:
        raise ValueError(f"{name}: each class needs at least 10 rows (target '{tcol}').")

    feats = {}
    for col in df.columns:
        if col == tcol or col in fel.IGNORED_COLUMNS or col.startswith("unnamed"):
            continue
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        if s.notna().mean() > 0.5 and num.notna().sum() >= 0.9 * s.notna().sum():
            x = num.astype(float)
        else:
            if s.nunique(dropna=True) > 20 or s.notna().mean() <= 0.5:
                continue
            cats = sorted(s.dropna().astype(str).unique())
            codes = pd.Categorical(s.astype(str).where(s.notna()), categories=cats).codes
            x = pd.Series(codes, index=s.index).astype(float).where(codes != -1)
        if x.nunique(dropna=True) < 2:
            continue
        canon = alias_lookup().get(col, col)
        if canon not in feats:
            feats[canon] = x
    X = pd.DataFrame(feats)
    if X.shape[1] < 2:
        raise ValueError(f"{name}: fewer than 2 usable feature columns after cleaning.")
    return X, y, tcol, mapping


def _check_client(name, y):
    if min(np.sum(y == 0), np.sum(y == 1)) < 10:
        raise ValueError(f"{name}: each class needs at least 10 rows (has {np.sum(y == 0)} / {np.sum(y == 1)}).")


def load_datasets(paths, target, generic, simulate_k, max_rows, seed=0):
    """Returns (native, features, mode). native[name] = (X, y, target_col, mapping)."""
    native, mode = {}, "strict"
    if not generic:
        try:
            for n, p in paths.items():
                native[n] = fel.load_client_csv(p, n)
        except Exception as e:
            print(f"\n[strict clinical loader failed: {str(e).strip()[:200]}]\n-> using generic loader")
            native, generic = {}, True
    if generic:
        mode = "generic"
        for n, p in paths.items():
            native[n] = load_generic_csv(p, n, target)
            print(f"{n}: rows={len(native[n][1])} cols={native[n][0].shape[1]} "
                  f"target={native[n][2]!r} pos_rate={native[n][1].mean():.3f}")

    rng = np.random.default_rng(seed)
    if len(native) == 1:
        if simulate_k < 2:
            raise ValueError("Only one dataset given: upload at least 2 CSVs (one per site) "
                             "or use --simulate_clients K to split it into K simulated sites.")
        (name, (X, y, tc, mp)), = native.items()
        parts = [[] for _ in range(simulate_k)]
        for cls in (0, 1):
            ids = rng.permutation(np.where(y == cls)[0])
            for i, chunk in enumerate(np.array_split(ids, simulate_k)):
                parts[i].extend(chunk.tolist())
        native = {f"Site_{i + 1}": (X.iloc[sorted(pt)].reset_index(drop=True), y[sorted(pt)], tc, mp)
                  for i, pt in enumerate(parts)}

    if max_rows and max_rows > 0:
        for n, (X, y, tc, mp) in list(native.items()):
            if len(y) > max_rows:
                idx = np.sort(rng.choice(len(y), max_rows, replace=False))
                native[n] = (X.iloc[idx].reset_index(drop=True), y[idx], tc, mp)
    for n, (X, y, _, _) in native.items():
        _check_client(n, y)

    if mode == "strict":
        features = fel.build_global_feature_space(native)
    else:
        counts = {}
        for X, *_ in native.values():
            for c in X.columns:
                counts[c] = counts.get(c, 0) + 1
        features = sorted(counts, key=lambda c: (-counts[c], c))[:60]
        if len(features) < 3:
            raise ValueError("Fewer than 3 usable features across all files.")
        if len(native) > 1 and max(counts.values()) < 2:
            raise ValueError("The files share no common column names; sites must describe the same features.")
        shared = sum(1 for c in features if counts[c] == len(native))
        print(f"\n[GENERIC FEATURE SPACE] {len(features)} features ({shared} present at every site)")
    return native, features, mode


def pick_sensitive(native, features, requested):
    """A binary feature to use for the model-inversion attack."""
    if requested and requested.lower() != "auto":
        return fel.normalize_feature_name(requested)
    best, best_n = "", 0
    for f in features:
        n = sum(1 for X, *_ in native.values()
                if f in X.columns and X[f].dropna().nunique() == 2)
        if n > best_n:
            best, best_n = f, n
    return best


# ============================================================
# MAIN
# ============================================================

def main():
    global fel
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="fel_v19", help="module name of your V19 script")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--eps", type=float, nargs="+", default=[1, 5, 10, 50, 200, 1000])
    ap.add_argument("--dp_clip", type=float, default=5.0, help="per-block update clip norm for DP")
    ap.add_argument("--sensitive_feature", default="auto", help="hidden binary attribute for model inversion ('auto' = pick one)")
    ap.add_argument("--data", nargs="+", default=None, help="CSV files, as NAME=path or path (one per site). Default: V19's CLIENT_CSV_PATHS")
    ap.add_argument("--target", default=None, help="target column name (auto-detected if omitted)")
    ap.add_argument("--generic", action="store_true", help="skip the strict clinical loader; accept any tabular CSV")
    ap.add_argument("--simulate_clients", type=int, default=0, help="if one CSV is given, split it into K simulated sites")
    ap.add_argument("--max_rows", type=int, default=0, help="subsample each site to at most this many rows (0 = all)")
    ap.add_argument("--out", default="benchmark_outputs")
    ap.add_argument("--quick", action="store_true", help="1 seed, 10 rounds, 3 epsilons")
    args, _ = ap.parse_known_args()  # tolerant of Jupyter/Colab extra args
    if args.quick:
        args.seeds, args.rounds, args.eps = 1, 10, [5, 50, 500]

    fel = importlib.import_module(args.module)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 90)
    print("PRIVACY-PRESERVING HEALTHCARE AI BENCHMARK")
    print("=" * 90)
    if args.data:
        paths = {}
        for item in args.data:
            if "=" in item and not os.path.exists(item):
                n, p = item.split("=", 1)
            else:
                n, p = os.path.splitext(os.path.basename(item))[0], item
            paths[n] = p
    else:
        paths = dict(fel.CLIENT_CSV_PATHS)
    native, features, mode = load_datasets(paths, args.target, args.generic or bool(args.data and args.generic),
                                           args.simulate_clients, args.max_rows)
    fel.CLIENT_CSV_PATHS = {n: "" for n in native}   # V19 only uses the number of sites
    d = len(features)
    sens_feat = pick_sensitive(native, features, args.sensitive_feature)
    print(f"\n[DATA] mode={mode} sites={len(native)} features={d} sensitive_feature={sens_feat or 'none'}")
    with open(os.path.join(args.out, "dataset_info.json"), "w") as f:
        json.dump({"mode": mode, "features": features, "sensitive_feature": sens_feat,
                   "sites": [{"site": n, "rows": int(len(v[1])), "positive_rate": float(np.mean(v[1])),
                              "target": v[2], "features_present": int(v[0].shape[1])}
                             for n, v in native.items()]}, f, indent=2)
    configs = build_configs(args.eps)

    rows = []
    for seed in range(args.seeds):
        fel.SEED = seed
        np.random.seed(seed)
        clients = [fel.FELClient(cid, name, *native[name], features)
                   for cid, name in enumerate(native)]
        for idx, cfg in enumerate(configs):
            np.random.seed(seed * 1000 + idx)
            rng = np.random.default_rng(seed * 7919 + idx)
            if cfg["kind"] == "central":
                res = run_centralized(clients, d, args.rounds)
            else:
                res = run_federated(clients, d, args.rounds, cfg["dp"], cfg["eps"], cfg["secagg"], args.dp_clip)
            gm = res["gm"]
            m = gm.test_metrics(clients)
            mia = membership_inference(gm, clients, rng)
            inv = attribute_inference(gm, clients, features, sens_feat)
            if cfg["kind"] == "central":   # raw rows are shared: total exposure by construction
                leak = {"leak_cos": 1.0, "leak_relerr": 0.0, "leak_success": 1.0}
            else:
                leak = gradient_leakage(gm, clients, d, cfg["secagg"], cfg["dp"],
                                        res["sigma"], res["sens"], args.dp_clip, rng)
            row = {"config": cfg["name"], "family": cfg["family"], "seed": seed,
                   "epsilon": res["epsilon"], "rounds": res["rounds"],
                   "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                   "f1": m["f1"], "auc": m["auc"],
                   "train_time_s": res["train_time"], "comm_mb": res["comm_mb"],
                   **mia, **inv, **leak}
            rows.append(row)
            print(f"[seed {seed}] {cfg['name']:<30} acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
                  f"eps={res['epsilon']:.2f} t={res['train_time']:.1f}s comm={res['comm_mb']:.2f}MB "
                  f"MIAadv={mia['mia_adv']:.3f} inv_adv={inv['inv_adv']:.3f} leak_cos={leak['leak_cos']:.2f}")

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(os.path.join(args.out, "results_raw.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "results_summary.csv"))
    with open(os.path.join(args.out, "benchmark_report.json"), "w") as f:
        json.dump({"args": vars(args), "features": features,
                   "summary": summary.reset_index().to_dict("records")}, f, indent=2, default=str)

    show = ["family", "epsilon", "accuracy_mean", "f1_mean", "train_time_s_mean",
            "comm_mb_mean", "mia_adv_mean", "inv_adv_mean", "leak_cos_mean", "on_frontier"]
    print("\n" + "=" * 90 + "\nSUMMARY (mean over seeds)\n" + "=" * 90)
    print(summary[show].round(3).to_string())
    print("\n[SAVED]")
    print("  tables  :", os.path.abspath(args.out))
    make_plots(summary, args.out)


if __name__ == "__main__":
    main()
