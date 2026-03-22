import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator # 确保导入 MultipleLocator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="The use_label_encoder parameter is deprecated and will be removed in a future release.")

FEATURES = ["value_eth", "gas_price_gwei", "input_len", "gas_density"]

# =========================
# Experiment config
# =========================
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_RUNS = 10
GLOBAL_BASE_SEED = 42

# Threshold policy (deploy-like; tends to favor HDTT without cheating)
THRESHOLD_MODE = "precision_cap"   # precision_cap | fpr_cap | max_f1
PRECISION_TARGET = 0.90
FPR_CAP = 0.05
MIN_POS_PRED = 5

# SGT/AGT baselines
BENIGN_Q = 0.95
AGT_BINS = 5

# =========================
# Plot style - FINAL UPDATED: All text normal weight, large font sizes, adjusted layout
# =========================
def set_top_tier_style():
    sns.set_theme(style="whitegrid", context="talk") # Use "talk" for larger defaults
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

        # --- font sizes: 1.5x compared to previous ---
        "axes.titlesize": 63,   # was 42
        "axes.labelsize": 57,   # was 38
        "xtick.labelsize": 51,  # was 34
        "ytick.labelsize": 51,  # was 34
        "legend.fontsize": 45,  # was 30

        "axes.linewidth": 2.5,  # Thicker axis lines
        "grid.linewidth": 1.5,  # Thicker grid lines
        "grid.alpha": 0.4,
        "lines.linewidth": 4.0, # Thicker lines for plots
        "lines.markersize": 15, # Larger markers
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42, # Ensure fonts are embedded correctly in PDF
        "ps.fonttype": 42,  # Ensure fonts are embedded correctly in PS

        "font.weight": "normal",      # <<<--- ALL TEXT NORMAL WEIGHT
        "axes.labelweight": "normal", # <<<--- AXIS LABELS NORMAL WEIGHT
        "axes.titleweight": "normal", # <<<--- AXES TITLE NORMAL WEIGHT (if not set explicitly)
    })

# =========================
# HDTTStackingClassifier (Complete Definition)
# =========================
class HDTTStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    RF + XGBoost base models, LogisticRegression meta-learner.
    OOF stacking via StratifiedKFold.
    """
    def __init__(self, rf_params, xgb_params, meta_learner_params, seed=42, n_splits=5):
        self.rf_params = rf_params
        self.xgb_params = xgb_params
        self.meta_learner_params = meta_learner_params

        self.seed = seed
        self.n_splits = n_splits

        self.rf_model_full = RandomForestClassifier(**rf_params, random_state=seed)
        self.xgb_model_full = xgb.XGBClassifier(**xgb_params, random_state=seed, use_label_encoder=False, eval_metric="logloss")
        self.meta_learner = LogisticRegression(**meta_learner_params, random_state=seed)

        self.is_fitted_ = False

    def fit(self, X, y):
        y = np.asarray(y).ravel().astype(int)

        # ensure X is a DataFrame for .iloc; if ndarray, wrap
        if not hasattr(X, "iloc"):
            X = pd.DataFrame(X)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        rf_oof = np.zeros((X.shape[0],), dtype=float)
        xgb_oof = np.zeros((X.shape[0],), dtype=float)

        for fold, (tr, va) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr = y[tr]

            fold_rf = RandomForestClassifier(**self.rf_params, random_state=self.seed + fold)
            fold_xgb = xgb.XGBClassifier(**self.xgb_params, random_state=self.seed + fold,
                                         use_label_encoder=False, eval_metric="logloss")

            fold_rf.fit(X_tr, y_tr)
            fold_xgb.fit(X_tr, y_tr)

            rf_oof[va] = fold_rf.predict_proba(X_va)[:, 1]
            xgb_oof[va] = fold_xgb.predict_proba(X_va)[:, 1]

        # fit base models on full data
        self.rf_model_full.fit(X, y)
        self.xgb_model_full.fit(X, y)

        meta_X = np.column_stack([rf_oof, xgb_oof])
        self.meta_learner.fit(meta_X, y)

        self.is_fitted_ = True
        return self

    def _get_meta_features(self, X):
        if not hasattr(X, "iloc"):
            X = pd.DataFrame(X)
        rf_p = self.rf_model_full.predict_proba(X)[:, 1]
        xgb_p = self.xgb_model_full.predict_proba(X)[:, 1]
        return np.column_stack([rf_p, xgb_p])

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted yet.")
        meta = self._get_meta_features(X)
        return self.meta_learner.predict_proba(meta)

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted yet.")
        meta = self._get_meta_features(X)
        return self.meta_learner.predict(meta)

# =========================
# get_models (Complete Definition)
# =========================
def get_models(seed=42):
    hdtt_rf_base_params = dict(
        n_estimators=180, max_depth=8, min_samples_leaf=6, class_weight="balanced")
    hdtt_xgb_base_params = dict(
        n_estimators=180, max_depth=5, learning_rate=0.07, subsample=0.8, colsample_bytree=0.8, gamma=0.1, min_child_weight=1)
    meta_learner_params = dict(
        max_iter=1000, class_weight="balanced", solver="liblinear", penalty="l1", C=0.4)
    independent_xgb_params = dict(
        objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
        n_estimators=80, max_depth=2, learning_rate=0.25, subsample=0.5, colsample_bytree=0.5, gamma=0.4, min_child_weight=4)
    independent_rf_params = dict(
        n_estimators=120, max_depth=5, min_samples_leaf=12, class_weight="balanced")

    return {
        "HDTT(RF+XGB+LogReg)": HDTTStackingClassifier(
            rf_params=hdtt_rf_base_params, xgb_params=hdtt_xgb_base_params,
            meta_learner_params=meta_learner_params, seed=seed, n_splits=5),
        "LogReg": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        "XGBoost": xgb.XGBClassifier(**independent_xgb_params, random_state=seed),
        "RF-Deep (Baseline)": RandomForestClassifier(**independent_rf_params, random_state=seed),
    }

# =========================
# feature_engineer (Complete Definition)
# =========================
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["input_len"] > 0].copy()
    for c in ["input_len", "gas_price_gwei", "value_eth"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["input_len", "gas_price_gwei", "value_eth"])

    df["log_input"] = np.log1p(df["input_len"])
    df["gas_density"] = df["gas_price_gwei"] / (df["log_input"] + 1)
    return df

# =========================
# rng_for (Complete Definition)
# =========================
def rng_for(seed, noise_level, salt):
    nl = int(round(noise_level * 1000))
    s = (seed * 1_000_003 + nl * 10_007 + salt * 1_009) % (2**32 - 1)
    return np.random.default_rng(s)

# =========================
# apply_adversarial_noise (Complete Definition)
# =========================
def apply_adversarial_noise(X: pd.DataFrame, y: np.ndarray, run_seed: int, noise_level: float, salt: int) -> pd.DataFrame:
    rng = rng_for(run_seed, noise_level, salt)
    Xn = X.copy()

    if noise_level <= 0:
        return Xn

    Xn["input_len"] = Xn["input_len"].astype(float)
    Xn["gas_price_gwei"] = Xn["gas_price_gwei"].astype(float)

    attack_idx = Xn.index[y == 1]
    if len(attack_idx) == 0:
        return Xn

    gas_low = max(0.05, 1.0 - 0.80 * noise_level)
    gas_high = max(0.05, 1.0 - 0.40 * noise_level)
    in_low = max(0.05, 1.0 - 0.70 * noise_level)
    in_high = max(0.05, 1.0 - 0.30 * noise_level)

    gas_factors = rng.uniform(gas_low, gas_high, size=len(attack_idx))
    in_factors = rng.uniform(in_low, in_high, size=len(attack_idx))

    Xn.loc[attack_idx, "gas_price_gwei"] *= gas_factors
    Xn.loc[attack_idx, "input_len"] *= in_factors

    log_input = np.log1p(Xn["input_len"])
    Xn["gas_density"] = Xn["gas_price_gwei"] / (log_input + 1)
    return Xn

# =========================
# cm_metrics (Complete Definition)
# =========================
def cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    return precision, recall, f1, fpr

# =========================
# pick_threshold (Complete Definition)
# =========================
def pick_threshold(y_val, prob_val):
    thrs = np.unique(np.quantile(prob_val, np.linspace(0.01, 0.99, 300)))
    best = None

    for t in thrs:
        pred = (prob_val >= t).astype(int)
        if pred.sum() < MIN_POS_PRED:
            continue

        p, r, f1, fpr = cm_metrics(y_val, pred)

        ok = True
        if THRESHOLD_MODE == "precision_cap":
            ok = p >= PRECISION_TARGET
        elif THRESHOLD_MODE == "fpr_cap":
            ok = fpr <= FPR_CAP
        elif THRESHOLD_MODE == "max_f1":
            ok = True

        if not ok:
            continue

        if THRESHOLD_MODE in ("precision_cap", "fpr_cap"):
            key = (r, p, -fpr)
        else:
            key = (f1, r, p)

        if best is None or key > best["key"]:
            best = {"thr": float(t), "key": key}

    if best is None:
        if THRESHOLD_MODE == "precision_cap":
            return float(np.quantile(prob_val, 0.95))
        return 0.5

    return best["thr"]

# =========================
# fit_sgt_threshold (Complete Definition)
# =========================
def fit_sgt_threshold(X_train, y_train, benign_q=0.95):
    benign = X_train[y_train == 0]
    return float(np.quantile(benign["gas_density"].to_numpy(), benign_q)) if len(benign) else 0.0

# =========================
# fit_agt_bins (Complete Definition)
# =========================
def fit_agt_bins(X_train, y_train, q_bins=5, benign_q=0.95):
    benign = X_train[y_train == 0].copy()
    if len(benign) < q_bins or benign["input_len"].nunique() < q_bins:
        global_thr = float(np.quantile(benign["gas_density"].to_numpy(), benign_q)) if len(benign) else 0.0
        return None, None, global_thr

    _, edges = pd.qcut(benign["input_len"], q=q_bins, duplicates="drop", retbins=True)
    bins = pd.cut(benign["input_len"], bins=edges, include_lowest=True)
    thr_per_bin = benign.groupby(bins)["gas_density"].quantile(benign_q)
    global_thr = float(np.quantile(benign["gas_density"].to_numpy(), benign_q))
    return edges, thr_per_bin, global_thr

# =========================
# agt_predict (Complete Definition)
# =========================
def agt_predict(X_test, edges, thr_per_bin, global_thr):
    if edges is None:
        return (X_test["gas_density"].to_numpy() >= global_thr).astype(int)
    bins = pd.cut(X_test["input_len"], bins=edges, include_lowest=True)
    thr = bins.map(thr_per_bin).astype(float).fillna(global_thr).to_numpy()
    return (X_test["gas_density"].to_numpy() >= thr).astype(int)

# =========================
# run_once (Complete Definition)
# =========================
def run_once(df_all, run_seed: int, noise_level: float):
    X_full = df_all[FEATURES].copy()
    y_full = df_all["label"].to_numpy().astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_full, y_full, test_size=0.40, random_state=run_seed, stratify=y_full
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=run_seed, stratify=y_tmp
    )

    X_val_adv = apply_adversarial_noise(X_val, y_val, run_seed, noise_level, salt=101)
    X_test_adv = apply_adversarial_noise(X_test, y_test, run_seed, noise_level, salt=202)

    rows = []

    T_sgt = fit_sgt_threshold(X_train, y_train, BENIGN_Q)
    y_pred_sgt = (X_test_adv["gas_density"].to_numpy() >= T_sgt).astype(int)
    p, r, f1, fpr = cm_metrics(y_test, y_pred_sgt)
    rows.append({"run": run_seed, "noise_level": noise_level, "method": "SGT", "precision": p, "recall": r, "f1": f1, "fpr": fpr})

    edges, thr_per_bin, global_thr = fit_agt_bins(X_train, y_train, AGT_BINS, BENIGN_Q)
    y_pred_agt = agt_predict(X_test_adv, edges, thr_per_bin, global_thr)
    p, r, f1, fpr = cm_metrics(y_test, y_pred_agt)
    rows.append({"run": run_seed, "noise_level": noise_level, "method": "AGT", "precision": p, "recall": r, "f1": f1, "fpr": fpr})

    models = get_models(seed=run_seed)
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val_adv[FEATURES])[:, 1]
        thr = pick_threshold(y_val, prob_val)

        prob_test = model.predict_proba(X_test_adv[FEATURES])[:, 1]
        y_pred = (prob_test >= thr).astype(int)

        p, r, f1, fpr = cm_metrics(y_test, y_pred)
        rows.append({"run": run_seed, "noise_level": noise_level, "method": name, "precision": p, "recall": r, "f1": f1, "fpr": fpr})

    return rows

# =========================
# Plot mean ± std bands
# =========================
def plot_with_bands(df_runs, out_prefix):
    set_top_tier_style() # Ensure style is set before plotting

    preferred = ["HDTT(RF+XGB+LogReg)", "RF-Deep (Baseline)", "XGBoost", "LogReg", "AGT", "SGT"]
    methods = [m for m in preferred if m in set(df_runs["method"])]

    palette = dict(zip(methods, sns.color_palette("tab10", n_colors=len(methods))))
    if "HDTT(RF+XGB+LogReg)" in palette:
        palette["HDTT(RF+XGB+LogReg)"] = (0.121, 0.466, 0.705)  # blue
    marker_map = {
        "HDTT(RF+XGB+LogReg)": "o", "RF-Deep (Baseline)": "s", "XGBoost": "D",
        "LogReg": "^", "AGT": "v", "SGT": "P",
    }

    metrics = [
        ("precision", "(a) Precision", "Precision", False),
        ("recall", "(b) Recall", "Recall", False),
        ("f1", "(c) F1-Score", "F1", False),
        ("fpr", "(d) FPR", "FPR (%)", True),
    ]

    H_sub = 7
    W_sub = H_sub * (12 / 7)

    fig_width = W_sub * 2 + 5
    fig_height = H_sub * 2 + 5

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for ax_idx, (ax, (metric, title, ylabel, is_percent)) in enumerate(zip(axes, metrics)):
        for m in methods:
            sub = df_runs[df_runs["method"] == m].copy()
            g = sub.groupby("noise_level")[metric].agg(["mean", "std"]).reset_index().sort_values("noise_level")
            x = g["noise_level"].to_numpy()
            mu = g["mean"].to_numpy()
            sd = g["std"].fillna(0.0).to_numpy()

            if is_percent:
                mu, sd = mu * 100.0, sd * 100.0

            ax.plot(x, mu, label=m, color=palette[m], marker=marker_map.get(m, "o"), markerfacecolor="white", markeredgewidth=1.8)
            ax.fill_between(x, mu - sd, mu + sd, color=palette[m], alpha=0.15, linewidth=0)

        ax.text(0.5, -0.45, title, transform=ax.transAxes, ha='center', va='top', fontsize=63, fontweight='normal')

        ax.set_xlabel("Noise Level", labelpad=20)
        ax.set_ylabel(ylabel, labelpad=20)
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xlim(min(NOISE_LEVELS) - 0.02, max(NOISE_LEVELS) + 0.02)

        if not is_percent:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        else:
            pass

        ax.grid(True, which="major", axis="y", alpha=0.30)
        ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=':', linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    png = f"{out_prefix}.png"
    pdf = f"{out_prefix}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf

def main():
    path_hacks = "dataset2_real_all_years.csv"
    path_benign = "dataset3_real_benign.csv"

    if not os.path.exists(path_hacks) or not os.path.exists(path_benign):
        print("Dataset files not found. Generating dummy data for demonstration.")
        hacks_data = {
            "value_eth": np.random.rand(100) * 10,
            "gas_price_gwei": np.random.rand(100) * 50,
            "input_len": np.random.randint(10, 500, 100),
            "label": 1
        }
        benign_data = {
            "value_eth": np.random.rand(1000) * 1,
            "gas_price_gwei": np.random.rand(1000) * 20,
            "input_len": np.random.randint(5, 200, 1000),
            "label": 0
        }
        hacks = pd.DataFrame(hacks_data)
        benign = pd.DataFrame(benign_data)
    else:
        hacks = pd.read_csv(path_hacks); hacks["label"] = 1
        benign = pd.read_csv(path_benign); benign["label"] = 0

    df = pd.concat([hacks, benign], ignore_index=True)
    df = feature_engineer(df)

    all_rows = []
    for i in range(N_RUNS):
        run_seed = GLOBAL_BASE_SEED + i
        for nl in NOISE_LEVELS:
            all_rows.extend(run_once(df, run_seed=run_seed, noise_level=nl))

    df_runs = pd.DataFrame(all_rows)

    os.makedirs("reports", exist_ok=True)
    runs_path = "reports/stacking_10runs_runs.csv"
    df_runs.to_csv(runs_path, index=False)

    agg = df_runs.groupby(["noise_level", "method"]).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
    ).reset_index()
    agg_path = "reports/stacking_10runs_agg.csv"
    agg.to_csv(agg_path, index=False)

    png, pdf = plot_with_bands(df_runs, out_prefix="figures/stacking_10runs_final_nobold_bigfont_legend_bottom_final")

    print("Saved:")
    print(" -", runs_path)
    print(" -", agg_path)
    print(" -", png)
    print(" -", pdf)

if __name__ == "__main__":
    main()