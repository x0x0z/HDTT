import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import numpy as np

OUT_DIR = "new_version"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS_CANDIDATES = ["stacking_10runs_runs.csv", "reports/stacking_10runs_runs.csv"]

HDTT_NAME = "HDTT(RF+XGB+LogReg)"
METHOD_ORDER = [HDTT_NAME, "RF-Deep (Baseline)", "XGBoost", "LogReg", "AGT", "SGT"]

KNOWN_NOISE = 0.0
ZERODAY_NOISE = 0.5
FPR_PANEL_NOISE = 0.0  # keep as original setting

# ---------------- IEEE single-column friendly font sizes ----------------
# If you feel labels are still small/large in the final PDF, tweak these only.
BASE_FONT_SIZE = 15
TITLE_SIZE = 15
LABEL_SIZE = 15
TICK_SIZE = 13

PDF_DPI = 300
PNG_DPI = 300


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": BASE_FONT_SIZE,

        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,

        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,

        "figure.dpi": 100,
        "savefig.dpi": 100,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
    })


def load_runs(path):
    if path is None:
        print("CSV not found, generating dummy data...")
        data = []
        for m in METHOD_ORDER:
            base_score = np.random.uniform(0.7, 0.9)
            for r in range(10):
                val_recall = np.clip(base_score + np.random.uniform(-0.05, 0.05), 0, 1)
                val_fpr = np.clip(base_score * 0.1 + np.random.uniform(-0.01, 0.01), 0, 0.2)
                data.append({"run": r, "method": m, "noise_level": 0.0, "recall": val_recall, "fpr": val_fpr})
                data.append({"run": r, "method": m, "noise_level": 0.5, "recall": val_recall * 0.8, "fpr": val_fpr * 1.5})
        return pd.DataFrame(data)

    df = pd.read_csv(path)
    df["method"] = df["method"].astype(str)
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    for c in ["precision", "recall", "f1", "fpr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def methods_present(df_runs):
    present = sorted(set(df_runs["method"]))
    ordered = [m for m in METHOD_ORDER if m in present]
    rest = [m for m in present if m not in ordered]
    return ordered + rest


def palette(methods):
    cols = sns.color_palette("tab10", n_colors=max(6, len(methods)))
    pal = {m: cols[i % len(cols)] for i, m in enumerate(methods)}
    if HDTT_NAME in pal:
        pal[HDTT_NAME] = (0.121, 0.466, 0.705)  # blue
    return pal


def beautify_axis(ax, y_major=None, y_minor=None):
    ax.spines["top"].set_alpha(0.6)
    ax.spines["right"].set_alpha(0.6)
    ax.spines["bottom"].set_alpha(0.85)
    ax.spines["left"].set_alpha(0.85)

    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

    ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.6)
    ax.grid(True, which="minor", axis="y", alpha=0.18, linewidth=0.45)

    ax.tick_params(axis="both", which="both", width=0.8, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")


def short_name(m):
    if m == HDTT_NAME:
        return "HDTT"
    if m == "RF-Deep (Baseline)":
        return "RF-Deep"
    return m


def draw_box_and_points_neatly_aligned(
    ax,
    df_long,
    order_display,
    palette_display,
    panel_label,
    ylabel,
    ylim=None,
    y_major=None,
    y_minor=None,
    box_width=0.6,
    point_size=18,
    point_alpha=0.75,
    xtick_rotation=25,
    xtick_ha="right",
    xtick_pad=6,
    xtick_fontsize=13,
    panel_label_y=-0.20,
):
    sns.boxplot(
        data=df_long,
        x="method_display",
        y="value",
        order=order_display,
        palette=palette_display,
        width=box_width,
        linewidth=0.8,
        fliersize=0.0,
        ax=ax,
        boxprops=dict(alpha=0.85),
        medianprops=dict(color="#333333", linewidth=1.1),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    spread_width = 0.35
    for i, method_name in enumerate(order_display):
        subset = df_long[df_long["method_display"] == method_name].copy().sort_values(by="run")
        count = len(subset)
        if count == 0:
            continue
        offsets = [0] if count == 1 else np.linspace(-spread_width / 2, spread_width / 2, count)
        ax.scatter(
            i + offsets,
            subset["value"].values,
            s=point_size,
            color=palette_display[method_name],
            alpha=point_alpha,
            edgecolors="white",
            linewidth=0.4,
            zorder=10
        )
    '''
    ax.set_title("")
    ax.text(
        0.5, panel_label_y, panel_label,
        transform=ax.transAxes,
        ha="center", va="top",
        fontweight="normal",
        fontsize=TITLE_SIZE
    )

    '''

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontweight="normal", labelpad=2)

    ax.set_xticks(range(len(order_display)))
    ax.set_xticklabels(order_display, rotation=xtick_rotation, ha=xtick_ha)
    ax.tick_params(axis="x", pad=xtick_pad)
    for t in ax.get_xticklabels():
        t.set_fontsize(xtick_fontsize)
        t.set_fontweight("normal")

    if ylim is not None:
        ax.set_ylim(ylim)

    beautify_axis(ax, y_major=y_major, y_minor=y_minor)


def main():
    set_style()

    runs_path = first_existing(RUNS_CANDIDATES)
    df_runs = load_runs(runs_path)

    methods = methods_present(df_runs)
    pal = palette(methods)

    display_name = {m: short_name(m) for m in methods}
    df_runs["method_display"] = df_runs["method"].map(display_name)

    order_display = [display_name[m] for m in methods]
    palette_display = {display_name[m]: pal[m] for m in methods}

    # ---- data ----
    known_long = df_runs[df_runs["noise_level"] == KNOWN_NOISE].copy()
    known_long["value"] = known_long["recall"]

    zero_long = df_runs[df_runs["noise_level"] == ZERODAY_NOISE].copy()
    zero_long["value"] = zero_long["recall"]

    fpr_long = df_runs[df_runs["noise_level"] == FPR_PANEL_NOISE].copy()
    fpr_long["value"] = fpr_long["fpr"]

    fpr_zero_long = df_runs[df_runs["noise_level"] == ZERODAY_NOISE].copy()
    fpr_zero_long["value"] = fpr_zero_long["fpr"]

    panels = [
        ("(a) Known Attacks", known_long, "Recall", (0.0, 1.05), 0.2, "panel_a_known_attacks"),
        ("(b) Zero-day Attacks", zero_long, "Recall", (0.0, 1.05), 0.2, "panel_b_zeroday_attacks"),
        ("(c) False Positive Rate", fpr_long, "FPR", None, 0.05, "panel_c_fpr"),
        ("(d) False Positive Rate (Zero-day)", fpr_zero_long, "FPR", None, 0.05, "panel_d_fpr_zeroday"),
    ]

    for panel_label, df_long, ylabel, ylim, y_major, stem in panels:
        # 3.5in wide is IEEE single-column width; height tuned for long x-ticks
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7))

        draw_box_and_points_neatly_aligned(
            ax, df_long, order_display, palette_display,
            panel_label=panel_label,
            ylabel=ylabel,
            ylim=ylim,
            y_major=y_major,
            y_minor=None,
            point_size=18,
            xtick_rotation=25,
            xtick_pad=6,
            xtick_fontsize=13,
            panel_label_y=-0.20,
        )

        plt.tight_layout(pad=0.2)
        fig.subplots_adjust(bottom=0.32)

        out_pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
        out_png = os.path.join(OUT_DIR, f"{stem}.png")
        fig.savefig(out_pdf, bbox_inches="tight", dpi=PDF_DPI)
        fig.savefig(out_png, bbox_inches="tight", dpi=PNG_DPI)
        plt.close(fig)

        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()