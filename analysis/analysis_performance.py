import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    if "2digit" in s: return "2 digit"
    if "3digit" in s: return "3 digit"
    if "4digit" in s and "+" not in s: return "4 digit"
    if "4+digit" in s or "4digit+" in s: return "4+ digit"
    if "1dp" in s: return "1dp"
    if "2dp" in s: return "2dp"
    return str(raw_val).strip()

def parse_dataset_features(data_dir="inference_eos_uni"):
    """Sweeps json logs and maps exact base_str_match, ml_str_match, and tk_str_match keys."""
    records = []
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0].replace('-Instruct','')
        if not model_name or model_name.endswith(".json"):
            model_name = "Qwen3-14B"
            
        if file_name.endswith("FERMAT_hard_question_fixed.json") and file_name.startswith("inference_data_"):
            format_type = "Worded_Question"
        elif file_name.endswith("FERMAT_hard.json") and file_name.startswith("inference_data_"):
            format_type = "Symbolic_Equation"
        else:
            continue
            
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
                if gold_digits is not None:
                    digit_count_label = f"{gold_digits}"
                else:
                    digit_count_label = "Unknown"
                
                metrics = entry.get("metrics", {})
                
                base_str_match = bool(metrics.get("base_str_match", False))
                ml_str_match = bool(metrics.get("ml_str_match", False))
                tk_str_match = bool(metrics.get("tk_str_match", False))
                
                # STRATEGY CATEGORIZATION FOR THE RECOVERY TRACKS
                if ml_str_match and tk_str_match and not base_str_match:
                    strategy_track = "ML & TK"
                elif ml_str_match and not base_str_match and not tk_str_match:
                    strategy_track = "ML only"
                elif not ml_str_match and not base_str_match and tk_str_match:
                    strategy_track = "TK only"
                elif not ml_str_match and not base_str_match and not tk_str_match:
                    strategy_track = "None"
                else:
                    strategy_track = "Other"
                    
                records.append({
                    "Model": model_name,
                    "Format": format_type,
                    "Subgroup": subgroup_label,
                    "Operation": operation,
                    "Digit_Count": digit_count_label,
                    "Base_Correct": base_str_match,
                    "Strategy_Track": strategy_track
                })

    return pd.DataFrame(records)

def label_sorting_key(label):
    """Generates an integer-based sorting key for digit count labels."""
    try:
        first_word = str(label).split()[0]
        return int(first_word)
    except (ValueError, IndexError):
        return str(label)

def plot_stacked_distribution_grid(df, model_name="Qwen3-14B"):
    """Generates a 2x3 grid with user-enforced structural ordering configurations."""
    model_df = df[df["Model"] == model_name].copy()
    if model_df.empty:
        print(f"No data discovered for model: {model_name}")
        return

    baseline_df = model_df[model_df["Base_Correct"] == True].copy()
    baseline_df["Strategy_Track"] = "Baseline"
    
    filtered_model_df = model_df[model_df["Strategy_Track"].isin(["None", "ML & TK", "ML only", "TK only"])]
    plot_df = pd.concat([filtered_model_df, baseline_df], ignore_index=True)
    
    # REVERSED SET ORDER SEQUENCE (matplotlib processes bottom-up, so reversing forces Baseline to top)
    strategy_order = ["None", "TK only", "ML only", "ML & TK", "Baseline"]
    
    # ENFORCED SUBGROUP CUSTOM ORDER SEQUENCE
    custom_subgroup_order = ["2 digit", "3 digit", "4 digit", "4+ digit", "1dp", "2dp"]
    
    features = ["Subgroup", "Operation", "Digit_Count"]
    feature_titles = ["Subgroup Categories", "Operation Categories", "Gold Digit Count Base"]
    formats = ["Symbolic_Equation", "Worded_Question"]
    format_titles = ["Symbolic Equation", "Worded Question"]
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 14.5)) 
    plt.subplots_adjust(hspace=0.65, wspace=0.18)
    
    collected_legend_data = []

    for feat_idx, feat in enumerate(features):
        if feat == "Subgroup":
            existing_subgroups = plot_df[feat].unique()
            unique_labels = [lbl for lbl in custom_subgroup_order if lbl in existing_subgroups]
            for lbl in existing_subgroups:
                if lbl not in unique_labels:
                    unique_labels.append(lbl)
        elif feat == "Digit_Count":
            unique_labels = sorted(plot_df[feat].unique(), key=label_sorting_key)
        else:
            unique_labels = sorted(plot_df[feat].unique(), key=lambda x: str(x))
            
        color_palette = sns.color_palette("tab10", len(unique_labels))
        color_map = dict(zip(unique_labels, color_palette))
        
        for fmt_idx, fmt in enumerate(formats):
            ax = axes[fmt_idx, feat_idx]
            slice_df = plot_df[plot_df["Format"] == fmt]
            
            if slice_df.empty:
                ax.text(0.5, 0.5, "No Data Vector", ha='center', va='center')
                continue
                
            ct = pd.crosstab(slice_df["Strategy_Track"], slice_df[feat])
            ct = ct.reindex(strategy_order).fillna(0)
            ct = ct.reindex(columns=unique_labels, fill_value=0)
            
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct = ct_pct.fillna(0)
            
            row_colors = [color_map[col] for col in ct_pct.columns]
            
            ct_pct.plot(
                kind="barh", 
                stacked=True, 
                ax=ax, 
                color=row_colors, 
                width=0.60, 
                edgecolor='white', 
                linewidth=0.5,
                legend=False
            )
            
            for i, col_name in enumerate(ct.columns):
                raw_counts = ct[col_name].astype(int).values
                container = ax.containers[i]
                labels = [f"{val}" if val > 0 else "" for val in raw_counts]
                
                ax.bar_label(
                    container, 
                    labels=labels, 
                    label_type='center', 
                    color='white', 
                    fontweight='bold', 
                    fontsize=10
                )
            
            ax.set_title(f"{feature_titles[feat_idx]} ({format_titles[fmt_idx]})", fontsize=13, fontweight='bold', pad=8)
            ax.set_xlabel("Distribution Share Inside Strategy Group (%)" if fmt_idx == 1 else "")
            ax.set_xlim(0, 100)
            ax.grid(True, axis='x', linestyle='-', linewidth=1.5, color='#e6e6e6')
            ax.grid(False, axis='y')
            
            if feat_idx == 0:
                ax.set_ylabel("Outcome Strategy Track", fontsize=12, fontweight='bold')
                ax.tick_params(axis='y', left=True, direction='out', length=6, width=1.5)
            elif feat_idx == 1:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis='y', left=False)
            elif feat_idx == 2:
                ax.set_ylabel("Outcome Strategy Track", fontsize=12, fontweight='bold')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(axis='y', right=True, direction='out', length=6, width=1.5)
            
        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in unique_labels]
        collected_legend_data.append((handles, unique_labels, feature_titles[feat_idx]))

    for i, (handles, labels, title) in enumerate(collected_legend_data):
        reference_ax = axes[0, i]
        bbox = reference_ax.get_position()
        
        graph_x = bbox.x0
        graph_w = bbox.width
        
        leg_ax = fig.add_axes([graph_x, 0.460, graph_w, 0.08]) 
        leg_ax.axis('off')
        
        if title == "Gold Digit Count Base":
            optimal_cols = int(np.ceil(len(labels) / 3))
        else:
            optimal_cols = len(labels)
            
        leg_ax.legend(
            handles, 
            labels, 
            title=title, 
            loc='center', 
            ncol=optimal_cols, 
            frameon=True,
            facecolor='white',
            edgecolor='#cccccc',
            fontsize=10,
            title_fontsize=11
        )
            
    plt.suptitle(f"{model_name} Model Analysis: Feature Distributions across Strategy Tracks\nMulti-Layer (ML) vs. Top-K (TK) Alignment Grid", y=0.97, fontsize=16, fontweight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_multi_layer_topk_grid_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Successfully generated top-down ordered distribution grid: {out_path}")

if __name__ == "__main__":
    master_df = parse_dataset_features("inference_eos_uni")
    if not master_df.empty:
        for model in master_df["Model"].unique():
            plot_stacked_distribution_grid(master_df, model_name=model)