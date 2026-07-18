import os
import json
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

import matplotlib.pyplot as plt
import seaborn as sns

def export_visual_results_panes(summary_report_data, data_dir="inference_eos_uni"):
    if not summary_report_data:
        return
        
    df = pd.DataFrame(summary_report_data)
    
    # Pre-process string representations back to raw floats
    for col in ["base_Bin_Acc", "Bin_Delta", "base_Digit_Acc", "Digit_Delta"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.replace("+", "").astype(float)
            
    df["McNemar_p"] = df["McNemar_p"].astype(float)
    df["Wilcoxon_p"] = df["Wilcoxon_p"].astype(float)

    sns.set_theme(style="whitegrid")
    scenarios = df["Scenario"].unique()
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(15, 7 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]
        
    for ax, scenario in zip(axes, scenarios):
        sub_df = df[df["Scenario"] == scenario]
        
        # Melt dataframe for seaborn grouped barplot orientation
        plot_df = pd.melt(
            sub_df, 
            id_vars=["Model", "Comparison", "McNemar_p", "Wilcoxon_p"], 
            value_vars=["Bin_Delta", "Digit_Delta"],
            var_name="Metric_Type", value_name="Delta_Value"
        )
        
        plot_df["Metric_Type"] = plot_df["Metric_Type"].map({
            "Bin_Delta": "Binary Match", 
            "Digit_Delta": "Granular Digit"
        })
        
        # Generate the main plot
        sns.barplot(
            data=plot_df, x="Model", y="Delta_Value", hue="Comparison",
            ax=ax, palette="muted", edgecolor="#444444", linewidth=1.0
        )
        
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"Net Performance Gain Profiles: {scenario.upper()}", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Improvement Margin Over Baseline ($\Delta$ %)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluated Model Architecture Family", fontsize=11, fontweight="bold")
        
        # --- THE SIGNIFICANCE LAYER MAPPER ---
        # Matches Matplotlib's layout coordinates to stamp significance flags
        for p in ax.patches:
            h = p.get_height()
            if np.isnan(h): continue
            
            # 1. Reverse-engineer which model and comparison pool this bar belongs to
            x_coord = p.get_x() + p.get_width() / 2.
            # Convert x coordinates to nearest categorical index matching the X-axis model list
            model_idx = int(round(p.get_x() + p.get_width() / 2. - 0.5 + (p.get_width() / 2.)))
            
            # Bound check indices against target matrix sizes
            models_list = sub_df["Model"].unique().tolist()
            if model_idx >= len(models_list) or model_idx < 0:
                continue
                
            model_name = models_list[model_idx]
            
            # Determine if this bar is binary or digit based on matplotlib patch color ordering
            # Seaborn plots hue cycles in strict rotation arrays
            # We can cross-match using the alpha / label configuration metadata
            # For a cleaner, direct approach, read the p-value corresponding to the matching row:
            # (Assuming standard hue distribution indexing)
            # To stay completely accurate, cross-reference the height value to its row:
            matched_rows = sub_df[(sub_df["Model"] == model_name)]
            
            if matched_rows.empty:
                continue
                
            # Pick p-value based on metric style
            # Binary checks McNemar, Digit checks Wilcoxon
            # A simple rule-of-thumb mapping based on bar iteration order:
            is_digit = "Digit" in str(p) or h in matched_rows["Digit_Delta"].values
            row = matched_rows.iloc[0]
            p_val = row["Wilcoxon_p"] if is_digit else row["McNemar_p"]
            
            # Determine significance label
            if p_val < 0.01:
                sig_text = "★★"  # Highly Significant
            elif p_val < 0.05:
                sig_text = "★"   # Significant
            else:
                sig_text = "ns"  # Not Significant (noise)
                
            # Place the significance annotation right above (or below) the value tag
            va_dir = 'bottom' if h >= 0 else 'top'
            offset = 12 if h >= 0 else -16
            
            # Draw numerical accuracy value text
            ax.annotate(f"{h:+.1f}%", (x_coord, h), ha='center', va=va_dir, 
                        fontsize=8, fontweight="bold", xytext=(0, 2 if h >= 0 else -10), textcoords='offset points')
            
            # Draw significance star marker
            ax.annotate(sig_text, (x_coord, h), ha='center', va=va_dir,
                        fontsize=9, fontweight="bold", color="darkgreen" if "★" in sig_text else "darkred",
                        xytext=(0, offset), textcoords='offset points')

        ax.legend(title="Intervention Context", loc="upper left")

    plt.tight_layout()
    chart_path = os.path.join(data_dir, "model_interventions_deltas.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
# =========================================================================
# 1. METRIC EXTRACTION AND DIGIT RATIO SCORER
# =========================================================================
def extract_and_score_metrics(entry, config_prefix):
    """Extracts binary match results and computes sequential digit accuracy."""
    metrics_block = entry.get("metrics", {})
    binary_key = f"{config_prefix}_str_match"
    is_binary_correct = bool(metrics_block.get(binary_key, False))
    
    gold_str = str(entry.get("gold_answer", "")).strip().replace(",", "")
    outputs_block = entry.get("outputs", {})
    
    if config_prefix == "base":
        gen_str = str(outputs_block.get("base_clean_numeric", ""))
    else:
        gen_str = str(outputs_block.get(f"{config_prefix}_generated_str", ""))
        
    norm_gen = gen_str.strip().lstrip(" ,.").replace(",", "")
    
    if not gold_str:
        return is_binary_correct, 0.0

    matches = 0
    min_len = min(len(gold_str), len(norm_gen))
    for i in range(min_len):
        if gold_str[i] == norm_gen[i]:
            matches += 1
            
    max_len = max(len(gold_str), len(norm_gen))
    digit_accuracy = (matches / max_len) if max_len > 0 else 0.0
    
    return is_binary_correct, digit_accuracy


# =========================================================================
# 2. CORE STATISTICAL TEST ROUTINE (WITH STATUS INDICATORS)
# =========================================================================
def execute_paired_tests(baseline_bin, oracle_bin, baseline_dig, oracle_dig, baseline_lbl, oracle_lbl, scenario_lbl, model_lbl):
    """Executes McNemar and Wilcoxon checks with strict visual signaling labels."""
    total_samples = len(baseline_bin)
    if total_samples == 0:
        return None

    # --- Descriptive Calculations ---
    base_bin_pct = np.mean(baseline_bin) * 100
    orac_bin_pct = np.mean(oracle_bin) * 100
    base_dig_avg = np.mean(baseline_dig) * 100
    orac_dig_avg = np.mean(oracle_dig) * 100

    # --- McNemar Contingency Processing ---
    a = sum(1 for b, o in zip(baseline_bin, oracle_bin) if b and o)
    b = sum(1 for b, o in zip(baseline_bin, oracle_bin) if b and not o)
    c = sum(1 for b, o in zip(baseline_bin, oracle_bin) if not b and o)
    d = sum(1 for b, o in zip(baseline_bin, oracle_bin) if not b and not o)

    if (b + c) == 0:
        mc_p = 1.0
        mc_sig = "🔴 NOT SIGNIFICANT"
    else:
        use_exact = (b + c < 25)
        mc_result = mcnemar([[a, b], [c, d]], exact=use_exact, correction=True)
        mc_p = mc_result.pvalue
        mc_sig = "🟢 SIGNIFICANT" if mc_p < 0.05 else "🔴 NOT SIGNIFICANT"

    # --- Wilcoxon Signed-Rank Processing ---
    differences = np.array(oracle_dig) - np.array(baseline_dig)
    non_zero_diffs = np.count_nonzero(differences)
    
    if non_zero_diffs == 0:
        wilc_p = 1.0
        wilc_sig = "🔴 NO VARIANCE"
    else:
        _, wilc_p = wilcoxon(oracle_dig, baseline_dig, alternative='greater')
        wilc_sig = "🟢 SIGNIFICANT" if wilc_p < 0.05 else "🔴 NOT SIGNIFICANT"

    # Pack results for structural datatable parsing
    return {
        "Model": model_lbl,
        "Scenario": scenario_lbl,
        "Comparison": f"{baseline_lbl.upper()} vs {oracle_lbl.upper()}",
        "N": total_samples,
        f"{baseline_lbl}_Bin_Acc": f"{base_bin_pct:.2f}%",
        f"{oracle_lbl}_Bin_Acc": f"{orac_bin_pct:.2f}%",
        "Bin_Delta": f"{orac_bin_pct - base_bin_pct:+.2f}%",
        "McNemar_p": f"{mc_p:.4e}",
        "McNemar_Sig": mc_sig,
        f"{baseline_lbl}_Digit_Acc": f"{base_dig_avg:.2f}%",
        f"{oracle_lbl}_Digit_Acc": f"{orac_dig_avg:.2f}%",
        "Digit_Delta": f"{orac_dig_avg - base_dig_avg:+.2f}%",
        "Wilcoxon_p": f"{wilc_p:.4e}",
        "Wilcoxon_Sig": wilc_sig
    }


# =========================================================================
# 3. INTERVENE, GROUP, AND EXTRACT PIPELINE
# =========================================================================
def run_granular_statistical_analysis(data_dir="inference_eos_uni"):
    scenarios = {
        "Worded Questions (Fixed Set)": os.path.join(data_dir, "inference_*_fixed.json"),
        "Symbolic Equations (Hard Set)": os.path.join(data_dir, "inference_*_hard.json")
    }
    
    strict_model_order = [
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", 
        "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"
    ]
    
    summary_report_data = []

    for scenario_name, path_pattern in scenarios.items():
        target_files = glob.glob(path_pattern)
        if not target_files:
            continue

        model_pools = {}

        for file_path in target_files:
            file_name = os.path.basename(file_path)
            model_name = file_name.replace("inference_data_", "").split("_")[0]
            if not model_name or model_name.endswith(".json"):
                model_name = "Unknown-Model"

            if model_name not in model_pools:
                model_pools[model_name] = {
                    "base_bin": [], "ml_bin": [], "tk_bin": [],
                    "base_dig": [], "ml_dig": [], "tk_dig": []
                }

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict): data = [data]
                except Exception:
                    continue

                for entry in data:
                    if "gold_answer" not in entry:
                        continue
                    
                    b_bin, b_dig = extract_and_score_metrics(entry, "base")
                    m_bin, m_dig = extract_and_score_metrics(entry, "ml")
                    t_bin, t_dig = extract_and_score_metrics(entry, "tk")

                    model_pools[model_name]["base_bin"].append(b_bin)
                    model_pools[model_name]["base_dig"].append(b_dig)
                    model_pools[model_name]["ml_bin"].append(m_bin)
                    model_pools[model_name]["ml_dig"].append(m_dig)
                    model_pools[model_name]["tk_bin"].append(t_bin)
                    model_pools[model_name]["tk_dig"].append(t_dig)

        # Enforce sorted parameter sizing structure
        sorted_models_present = [m for m in strict_model_order if m in model_pools]
        extra_models_found = [m for m in model_pools if m not in strict_model_order]
        final_processing_queue = sorted_models_present + extra_models_found

        for model in final_processing_queue:
            pools = model_pools[model]
            if not pools["base_bin"]:
                continue

            # Run Independent Pair A: BASE vs ML
            ml_res = execute_paired_tests(
                pools["base_bin"], pools["ml_bin"], pools["base_dig"], pools["ml_dig"],
                "base", "ml", scenario_name, model
            )
            if ml_res:
                summary_report_data.append(ml_res)

            # Run Independent Pair B: BASE vs TK
            tk_res = execute_paired_tests(
                pools["base_bin"], pools["tk_bin"], pools["base_dig"], pools["tk_dig"],
                "base", "tk", scenario_name, model
            )
            if tk_res:
                summary_report_data.append(tk_res)

    if not summary_report_data:
        print("⚠️ Matrix run complete, but zero tracking records were produced.")
        return

    df_report = pd.DataFrame(summary_report_data)
    csv_out = "granular_model_significance_report.csv"
    df_report.to_csv(csv_out, index=False)
    print(f"📋 Global analytical spreadsheet generated successfully: '{csv_out}'\n")

    # Terminal Log Summary Printer
    for scenario in df_report["Scenario"].unique():
        print("=" * 100)
        print(f" 📂 DATASET CONTEXT: {scenario.upper()}")
        print("=" * 100)
        
        sub_df = df_report[df_report["Scenario"] == scenario]
        for _, row in sub_df.iterrows():
            print(f"🤖 Model: {row['Model']:<13} | Test Track: {row['Comparison']}")
            print(f"  • Binary Match Margin : {row['base_Bin_Acc']} -> {row['ml_Bin_Acc' if 'ML' in row['Comparison'] else 'tk_Bin_Acc']} ({row['Bin_Delta']})")
            print(f"  • McNemar Significance: p = {row['McNemar_p']:<10} -> {row['McNemar_Sig']}")
            print(f"  • Digit Vector Margin : {row['base_Digit_Acc']} -> {row['ml_Digit_Acc' if 'ML' in row['Comparison'] else 'tk_Digit_Acc']} ({row['Digit_Delta']})")
            print(f"  • Wilcoxon Significance: p = {row['Wilcoxon_p']:<10} -> {row['Wilcoxon_Sig']}")
            print("-" * 100)
    if summary_report_data:
        # Run the visual renderer right before closing out the function execution
        export_visual_results_panes(summary_report_data, data_dir)


if __name__ == "__main__":
    run_granular_statistical_analysis()