import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Set clean scientific canvas layout defaults
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'figure.max_open_warning': 100, 
    'font.size': 11,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.5
})

OUTPUT_DIR = "inference_eos_uni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_subgroup_string(raw_val):
    """Normalizes variation in file logs to ensure precise alignment."""
    s = str(raw_val).lower().replace(" ", "").strip()
    if "2digit" in s: return "2 digit"
    if "3digit" in s: return "3 digit"
    if "4digit" in s and "+" not in s: return "4 digit"
    if "4+digit" in s or "4digit+" in s: return "4+ digit"
    if "1dp" in s: return "1dp"
    if "2dp" in s: return "2dp"
    return str(raw_val).strip()

def parse_dataset_features(data_dir="inference_eos_uni"):
    """Sweeps json logs and filters strictly for Symbolic Equation formats."""
    records = []
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0].replace('-Instruct','')
        if not model_name or model_name.endswith(".json"):
            model_name = "Qwen3-14B"
            
        # Isolate Symbolic Equation format logs exclusively
        if file_name.endswith("FERMAT_hard.json") and file_name.startswith("inference_data_"):
            format_type = "Symbolic_Equation"
        else:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for entry in data:
                subgroup_label = clean_subgroup_string(entry.get("subgroup", "Unknown"))
                operation = str(entry.get("operation", "Unknown")).strip().capitalize()
                
                gold_digits = entry.get("gold_digit_count", None)
                digit_count_label = f"{gold_digits}" if gold_digits is not None else "Unknown"
                
                metrics = entry.get("metrics", {})
                base_str_match = bool(metrics.get("base_str_match", False))
                ml_str_match = bool(metrics.get("ml_str_match", False))
                tk_str_match = bool(metrics.get("tk_str_match", False))
                
                if base_str_match:
                    outcome_group = "Baseline"
                elif ml_str_match or tk_str_match:
                    outcome_group = "Recovery"
                else:
                    outcome_group = "None"
                    
                records.append({
                    "Model": model_name,
                    "Format": format_type,
                    "Subgroup": subgroup_label,
                    "Operation": operation,
                    "Digit_Count": digit_count_label,
                    "Outcome_Group": outcome_group
                })

    return pd.DataFrame(records)

def label_sorting_key(label):
    """Generates sorting keys for digit counts."""
    try:
        return int(str(label).split()[0])
    except (ValueError, IndexError):
        return str(label)

def plot_global_features_row_legend_bottom(df):
    """
    Generates a 1x3 grid representing the global dataset average performance profile:
      - Columns: Subgroup, Operation, Digit Count
      - Graph type: Stacked vertical bar chart (Baseline -> Recovery -> None)
      - Key / Legend: Positioned below the plots.
    """
    if df.empty:
        print("No Symbolic Equation data discovered to plot.")
        return

    outcome_order = ["Baseline", "Recovery", "None"]
    outcome_colors = {
        "Baseline": "#2ca02c",  # Green (Bottom)
        "Recovery": "#ff7f0e",  # Orange (Middle)
        "None": "#d62728"      # Red (Top)
    }

    features = ["Subgroup", "Operation", "Digit_Count"]
    feature_titles = ["Subgroups", "Operations", "Gold Digit Counts"]
    
    # Establish canonical ordering for X-axes
    custom_subgroup_order = ["2 digit", "3 digit", "4 digit", "4+ digit", "1dp", "2dp"]
    
    # Initialize a 1x3 subplot setup with expanded bottom spacing for the legend
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)
    plt.subplots_adjust(wspace=0.15, bottom=0.18, top=0.82)

    for col_idx, feat in enumerate(features):
        ax = axes[col_idx]
        
        # Identify valid x-ticks
        if feat == "Subgroup":
            existing = df[feat].unique()
            x_categories = [lbl for lbl in custom_subgroup_order if lbl in existing]
            for lbl in existing:
                if lbl not in x_categories:
                    x_categories.append(lbl)
        elif feat == "Digit_Count":
            x_categories = sorted(df[feat].unique(), key=label_sorting_key)
        else:
            x_categories = sorted(df[feat].unique())

        # Crosstab and normalize to %
        ct = pd.crosstab(df[feat], df["Outcome_Group"])
        ct = ct.reindex(x_categories).fillna(0).reindex(columns=outcome_order, fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct = ct_pct.fillna(0)

        # Draw the stacked bar chart
        ct_pct.plot(
            kind="bar", 
            stacked=True, 
            ax=ax, 
            width=0.55,
            color=[outcome_colors[o] for o in outcome_order], 
            edgecolor='white', 
            linewidth=0.5, 
            legend=False
        )

        # Inside-bar annotations: Raw Count + %
        for bar_idx, category in enumerate(x_categories):
            cum_sum = 0.0
            for outcome in outcome_order:
                val = ct_pct.loc[category, outcome]
                raw_count = int(ct.loc[category, outcome])
                if val > 5.0:  # Avoid crowding tiny bars
                    ax.text(
                        bar_idx, 
                        cum_sum + (val / 2.0), 
                        f"{raw_count}\n({val:.0f}%)", 
                        ha='center', 
                        va='center', 
                        color='white', 
                        fontweight='bold', 
                        fontsize=10
                    )
                cum_sum += val

        # Subplot Aesthetics
        # ax.set_title(feature_titles[col_idx], fontsize=14, fontweight='bold', pad=12)
        
        # --- CONDITIONAL X-AXIS ROTATION ---
        # Keep Gold Digit Count labels upright (rotation=0) and centered
        if feat == "Digit_Count":
            ax.set_xticklabels(x_categories, rotation=0, ha='center')
        else:
            ax.set_xticklabels(x_categories, rotation=25, ha='right')
            
        ax.set_ylabel("Outcome Cumulative Share (%)" if col_idx == 0 else "")
        ax.set_xlabel(feature_titles[col_idx], fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylim(0, 105)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, color='#999999')

    # --- Unified Legend Positioned nicely at the bottom ---
    legend_elements = [
        Line2D([0], [0], color=outcome_colors["None"], lw=6, label="None (Failed Completely)"),
        Line2D([0], [0], color=outcome_colors["Recovery"], lw=6, label="Recovery (Initially Failed, Saved by ML/TK)"),
        Line2D([0], [0], color=outcome_colors["Baseline"], lw=6, label="Baseline (Initially Correct)")
    ]
    
    # bbox_to_anchor=(0.5, 0.08) places the legend neatly below the charts and tick labels
    fig.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.92), 
        ncol=3,
        frameon=True, 
        facecolor='white', 
        edgecolor='#cccccc', 
        fontsize=12, 
        title="Path Outcomes (Bottom-to-Top Stack)"
    )
    
    plt.suptitle(
        "Mean Model Performance Distribution for Symbolic Equations",
        y=0.96, fontsize=17, fontweight='bold'
    )
    
    out_path = os.path.join(OUTPUT_DIR, "symbolic_equations_global_stacked_row.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 Complete! 1x3 Global stacked average grid (with bottom key) generated at: {out_path}\n")

if __name__ == "__main__":
    master_df = parse_dataset_features("inference_eos_uni")
    plot_global_features_row_legend_bottom(master_df)