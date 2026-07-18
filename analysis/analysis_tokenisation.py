import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Set clean scientific canvas layout defaults
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'figure.max_open_warning': 100, 
    'font.size': 12,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.5
})

OUTPUT_DIR = "inference_eos_uni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_subgroup_string(raw_val):
    """Normalizes variation in file logs to ensure precise alignment."""
    s = str(raw_val).lower().replace(" ", "").strip()
    if "2digit" in s: return "2 Digit"
    if "3digit" in s: return "3 Digit"
    if "4digit" in s and "+" not in s: return "4 Digit"
    if "4+digit" in s or "4digit+" in s: return "4+ Digit"
    if "1dp" in s: return "1DP"
    if "2dp" in s: return "2DP"
    return str(raw_val).strip().capitalize()

def normalize_model_name(raw_name):
    """Maps varied file strings to strict target cohort keys to avoid missing data."""
    name_lower = raw_name.lower()
    if "qwen" in name_lower and "4b" in name_lower:
        return "Qwen3-4B"
    if "qwen" in name_lower and "8b" in name_lower:
        return "Qwen-8B"
    if "llama" in name_lower and "3.2" in name_lower and "3b" in name_lower:
        return "Llama-3.2-3B-Instruct"
    if "llama" in name_lower and "3.1" in name_lower and "8b" in name_lower:
        return "llama-3.1-8B"
    return raw_name

def parse_tokenization_features(data_dir="."):
    """Sweeps json logs to extract gold digit counts, operational accuracy, and subgroups."""
    records = []
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        raw_model_name = file_name.replace("inference_data_", "").split("_")[0]
        if not raw_model_name or raw_model_name.endswith(".json"):
            continue
            
        model_name = normalize_model_name(raw_model_name)
        format_type = "Symbolic_Equation" if "hard.json" in file_name else "Worded_Question"
            
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except json.JSONDecodeError:
                continue

            for entry in data:
                subgroup_label = clean_subgroup_string(entry.get("subgroup", "Unknown"))
                operation = str(entry.get("operation", "Unknown")).strip().capitalize()
                
                gold_digits = entry.get("gold_digit_count", None)
                if gold_digits is None:
                    continue
                
                metrics = entry.get("metrics", {})
                base_str_match = bool(metrics.get("base_str_match", False))
                
                records.append({
                    "Model": model_name,
                    "Format": format_type,
                    "Subgroup": subgroup_label,
                    "Operation": operation,
                    "Gold_Digit_Count": int(gold_digits),
                    "Accuracy": 1 if base_str_match else 0
                })

    return pd.DataFrame(records)

def generate_unified_grid(df):
    """Generates a strict 2-row (Formats) x 3-column (Subgroups) alignment grid

    plotting all models simultaneously with custom scale colors and style curves.
    """
    # Enforced Column Sorting Sequence
    target_subgroups = ["2 Digit", "4 Digit", "1DP"]
    # Enforced Row Sorting Sequence 
    target_formats = ["Symbolic_Equation", "Worded_Question"]
    target_operation = "Division"
    
    # Strict Model Styling Maps
    model_configs = {
        "Qwen-8B": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "llama-3.1-8B": {"color": "#1f77b4", "linestyle": "--", "marker": "s"},
        "Qwen3-4B": {"color": "#d62728", "linestyle": "-", "marker": "o"},
        "Llama-3.2-3B-Instruct": {"color": "#d62728", "linestyle": "--", "marker": "s"}
    }
    all_models = list(model_configs.keys())
    
    df_filtered = df[
        (df["Model"].isin(all_models)) & 
        (df["Subgroup"].isin(target_subgroups)) & 
        (df["Operation"] == target_operation)
    ].copy()
    
    if df_filtered.empty:
        print(" -> Error: No execution records matching structural constraints discoverable.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True, sharex=False)
    plt.subplots_adjust(top=0.85, wspace=0.12, hspace=0.35)
    
    for r_idx, fmt in enumerate(target_formats):
        for c_idx, subgroup in enumerate(target_subgroups):
            ax = axes[r_idx, c_idx]
            
            slice_df = df_filtered[(df_filtered["Format"] == fmt) & (df_filtered["Subgroup"] == subgroup)]
            
            if slice_df.empty:
                ax.text(0.5, 0.5, "No Data Vector", ha='center', va='center', fontsize=11, style='italic')
                ax.set_title(f"{subgroup}", fontsize=13, fontweight='bold')
                continue
                
            summary = slice_df.groupby(["Model", "Gold_Digit_Count"])["Accuracy"].mean().reset_index()
            summary["Accuracy_Pct"] = summary["Accuracy"] * 100
            
            # Trace individual model curves concurrently on this canvas panel
            for model_name in all_models:
                m_subset = summary[summary["Model"] == model_name]
                if m_subset.empty:
                    continue
                    
                cfg = model_configs[model_name]
                ax.plot(
                    m_subset["Gold_Digit_Count"],
                    m_subset["Accuracy_Pct"],
                    color=cfg["color"],
                    linestyle=cfg["linestyle"],
                    marker=cfg["marker"],
                    linewidth=3,
                    markersize=8,
                    alpha=0.85
                )
            
            # Subplot Titles incorporating specific Column Labels
            ax.set_title(f"{subgroup} ({'Symbolic' if r_idx == 0 else 'Worded'})", fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel("Gold Digit Count", fontsize=11, fontweight='bold')
            ax.set_ylim(-5, 105)
            
            unique_ticks = sorted(slice_df["Gold_Digit_Count"].unique())
            ax.set_xticks(unique_ticks)
            ax.grid(True, linestyle=':', alpha=0.6, color='#999999')
            
            if c_idx == 0:
                ax.set_ylabel("Baseline Accuracy (%)", fontsize=11, fontweight='bold')
                
    # ---- Build Cleaned Legend System (Evaluated Models Only) ----
    legend_elements = []
    for m_name, cfg in model_configs.items():
        m_name_map = {"Qwen3-4B":"Qwen3-4B", "Qwen-8B":"Qwen3-8B", "Llama-3.2-3B-Instruct":"Llama-3.2-3B", "llama-3.1-8B":"Llama-3.1-8B"}
        legend_elements.append(Line2D(
            [0], [0], 
            color=cfg["color"], 
            linestyle=cfg["linestyle"], 
            marker=cfg["marker"], 
            lw=3, 
            markersize=8,
            label=f"{m_name_map[m_name]}"
        ))
        
    fig.legend(
        handles=legend_elements, 
        loc="lower center",          # Anchors the bottom-center of the legend box
        bbox_to_anchor=(0.5, 0.89),   # Positioned horizontally centered (0.5) and vertically right below the suptitle
        ncol=len(legend_elements),   # Forces all model items to align side-by-side horizontally
        frameon=True, 
        facecolor="white", 
        edgecolor="#cccccc", 
        fontsize=12,
        title="Evaluated Models",
        title_fontsize=13
    )
    
    plt.suptitle("Performance Across Gold Digit Count for Division", y=0.98, fontsize=17, fontweight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, "division_cross_model_structured_matrix_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Exported Cross-Model Matrix Grid to: {out_path}")

if __name__ == "__main__":
    print("Beginning multi-model matrix execution loop...")
    master_df = parse_tokenization_features("inference_eos_uni")
    if not master_df.empty:
        generate_unified_grid(master_df)