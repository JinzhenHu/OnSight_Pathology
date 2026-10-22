"""Fig 5d"""
import argparse
import os

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

MAX_PATCHES = 20
MIN_PATCHES = 1
ITERATIONS = 100
BASE_SEED = 987654321
FEATURE_NAME = "Nucleus.Intensity.Mean_median"

def load_patch_tables(df_survival, patch_root):
    metrics_cache, missing = {}, 0
    for idx, row in df_survival.iterrows():
        folder_name = row["PATIENT_ID"].split("-")[-1]
        metrics_file = os.path.join(patch_root, folder_name, "aggregate_metrics.csv")
        if os.path.exists(metrics_file):
            metrics_cache[idx] = pd.read_csv(metrics_file).iloc[:-1]
        else:
            missing += 1
    print(f"Loaded survival patients: {len(df_survival)}")
    print(f"Metrics files found: {len(metrics_cache)} | missing: {missing}")
    return metrics_cache

def run_simulation(df_survival, metrics_cache):
    results, source_data_rows = [], []
    for k in range(MAX_PATCHES, MIN_PATCHES - 1, -1):
        p_values_for_k = []
        for it in range(ITERATIONS):
            patient_values, valid_indices = [], []
            for idx, df_metrics in metrics_cache.items():
                if len(df_metrics) < k:
                    continue
                seed = BASE_SEED + k * 1000 + it + idx
                sampled = df_metrics.sample(n=k, replace=False, random_state=seed)
                patient_values.append(sampled[FEATURE_NAME].mean())
                valid_indices.append(idx)
            if len(valid_indices) < 10:
                continue

            df_current = df_survival.loc[valid_indices].copy()
            df_current["Agg_Feature"] = patient_values
            median_value = df_current["Agg_Feature"].median()
            df_current["Group"] = np.where(df_current["Agg_Feature"] >= median_value, "High", "Low")

            T = df_current["OS_MONTHS"] / 12.0
            E = df_current["IS_DECEASED"].astype(int)
            ix_high = df_current["Group"] == "High"
            res = logrank_test(T[ix_high], T[~ix_high], event_observed_A=E[ix_high], event_observed_B=E[~ix_high])
            p_value = float(res.p_value)
            p_values_for_k.append(p_value)
            source_data_rows.append({
                "Num_Patches": int(k), "Iteration": int(it + 1), "Logrank_P_Value": p_value,
                "N_Patients": int(len(df_current)), "N_Events": int(E.sum()),
                "Median_Feature_Threshold": float(median_value), "N_High": int(ix_high.sum()),
                "N_Low": int((~ix_high).sum()), "Feature": FEATURE_NAME,
                "Aggregation": "Mean across sampled patches", "Base_Seed": int(BASE_SEED)})

        if p_values_for_k:
            results.append({
                "Num_Patches": k,
                "Mean_P_Value": float(np.mean(p_values_for_k)),
                "Median_P_Value": float(np.median(p_values_for_k)),
                "P_Value_25": float(np.percentile(p_values_for_k, 25)),
                "P_Value_75": float(np.percentile(p_values_for_k, 75)),
                "N_Valid_Iters": len(p_values_for_k)})
            print(f"k={k:2d} | median p={np.median(p_values_for_k):.4f} | iters={len(p_values_for_k)}")
        else:
            print(f"k={k:2d} | no valid iterations")

    df_results = pd.DataFrame(results).sort_values("Num_Patches", ascending=False)
    df_source = pd.DataFrame(source_data_rows).sort_values(["Num_Patches", "Iteration"], ascending=[False, True])
    return df_results, df_source

def plot_results(df_results, out_png):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "Arial", "font.size": 11,
                         "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                         "xtick.major.size": 4, "ytick.major.size": 4})
    d = df_results.sort_values("Num_Patches", ascending=False)
    x = d["Num_Patches"].values
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.fill_between(x, d["P_Value_25"], d["P_Value_75"], color="#308083", alpha=0.18, linewidth=0,
                    label="Interquartile range")
    ax.plot(x, d["Median_P_Value"], color="#308083", linewidth=2.2, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.4, label="Median p-value")
    ax.plot(x, d["Mean_P_Value"], color="#E41A1C", linewidth=2.2, linestyle="--", marker="s", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.4, label="Mean p-value")
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of randomly sampled patches per patient", labelpad=8, fontsize=16)
    ax.set_ylabel("Log Rank P-value", labelpad=8, fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_ylim(0, max(d["P_Value_75"].max(), d["Mean_P_Value"].max()) * 1.08)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.9, 1.0), fontsize=18, handlelength=2.5)
    ax.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survival-csv", required=True)
    ap.add_argument("--patch-root", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--plot", action="store_true", help="also draw the Fig. 5d panel")
    args = ap.parse_args()

    df_survival = pd.read_csv(args.survival_csv)
    metrics_cache = load_patch_tables(df_survival, args.patch_root)
    df_results, df_source = run_simulation(df_survival, metrics_cache)

    os.makedirs(args.out_dir, exist_ok=True)
    df_source.to_csv(os.path.join(args.out_dir, "Source_Data_Fig5d_Patch_Dilution.csv"), index=False)
    df_results.to_csv(os.path.join(args.out_dir, "Source_Data_Fig5d_Summary.csv"), index=False)
    print("Saved:", os.path.join(args.out_dir, "Source_Data_Fig5d_Patch_Dilution.csv"))
    if args.plot:
        plot_results(df_results, os.path.join(args.out_dir, "pvalue_vs_patches_publication.png"))

if __name__ == "__main__":
    main()
