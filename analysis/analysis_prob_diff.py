import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Set clean scientific canvas defaults
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'figure.max_open_warning': 100, 
    'font.size': 12,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.5
})

OUTPUT_DIR = "inference_eos_uni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_margin_metrics(data_dir="inference_eos_uni"):
    """Extracts step-0 layer metrics, isolating both the Top-1/Top-2 margin
       and the specific Chosen vs. Gold token discrepancy for incorrect tracking paths.
    """
    records = []
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0]
        if not model_name or model_name.endswith(".json"):
            model_name = "Qwen3-14B"
            
        format_type = "Worded_Question" if "_question_fixed.json" in file_name else "Symbolic_Equation"
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for entry in data:
                tokens_block = entry.get("tokens", {})
                ranks_list = tokens_block.get("tk_chosen_vocabulary_ranks", [])
                
                is_token_correct = bool(ranks_list and ranks_list[0] == 0)
                accuracy_status = "Correct" if is_token_correct else "Incorrect"
                
                layer_traces = entry.get("layer_traces", {})
                fine_grain = layer_traces.get("ml_layer_predictions_fine_grain", [])
                
                valid_layers_per_step = layer_traces.get("ml_valid_layers_per_step", [])
                step_0_valid_layers = valid_layers_per_step[0] if valid_layers_per_step else []
                
                if not fine_grain or len(fine_grain) == 0:
                    continue
                
                first_step_layers = fine_grain[0]
                for layer_key, layer_data in first_step_layers.items():
                    try:
                        layer_idx = int(''.join(filter(str.isdigit, layer_key)))
                        top_2_list = layer_data.get("absolute_top_2_predictions", [])
                        gold_tracking = layer_data.get("gold_token_tracking", {})
                        
                        p1 = float(top_2_list[0].get("prob", 0.0)) if len(top_2_list) >= 1 else 0.0
                        p2 = float(top_2_list[1].get("prob", 0.0)) if len(top_2_list) >= 2 else 0.0
                        prob_diff = abs(p1 - p2)
                        
                        p_gold = float(gold_tracking.get("prob", 0.0))
                        chosen_vs_gold_diff = abs(p1 - p_gold) if not is_token_correct else 0.0
                        
                        is_layer_correct = 1 if layer_idx in step_0_valid_layers else 0
                        
                        records.append({
                            "Model": model_name,
                            "Format": format_type,
                            "Layer": layer_idx,
                            "Prob_Diff": prob_diff,
                            "Chosen_Gold_Diff": chosen_vs_gold_diff,
                            "Status": accuracy_status,
                            "Layer_Correct": is_layer_correct
                        })
                    except (ValueError, TypeError):
                        continue

    return pd.DataFrame(records)

def plot_cross_model_worded_margin_grid(df, num_last_layers=15):
    """Plots Worded Question performance comparing Qwen3-8B (Left) against 
       Llama-3.1-8B-Instruct (Right) with independent model layer x-axes,
       uncoupled custom-fitted logarithmic y-axes, increased layout spacing,
       and a unified legend placed below the charts.
    """
    target_models = ["Qwen3-8B", "Llama-3.1-8B"]
    
    filtered_df = df[(df["Model"].isin(target_models)) & 
                     (df["Format"] == "Symbolic_Equation") & 
                     (df["Layer"] > 0)].copy()
                     
    if filtered_df.empty:
        print("No matching data found for Qwen3-8B or Llama-3.1-8B-Instruct in Symbolic Equation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    # Increased wspace from 0.22 to 0.35 to separate the subgraphs more significantly
    plt.subplots_adjust(wspace=0.35, bottom=0.20) 
    
    colors = {"Correct": "#2ca02c", "Incorrect": "#d62728"}
    
    for idx, model in enumerate(target_models):
        ax_margin = axes[idx]
        model_df = filtered_df[filtered_df["Model"] == model].copy()
        
        if model_df.empty:
            ax_margin.text(0.5, 0.5, f"Missing Data for {model}", ha='center', va='center')
            continue
            
        # Dynamically scale the final layers specifically to the current loop's model layout depth
        max_layer = model_df["Layer"].max()
        min_layer_cutoff = (max_layer - num_last_layers) + 1
        model_df = model_df[model_df["Layer"] >= min_layer_cutoff].copy()
        
        # --- LOCAL DYNAMIC Y-LIMIT DETECTOR PER MODEL ---
        observed_min_diff = 1.0
        sum_check = model_df.groupby(["Layer", "Status"]).mean(numeric_only=True).reset_index()
        for col in ["Prob_Diff", "Chosen_Gold_Diff"]:
            if col in sum_check.columns:
                valid_vals = sum_check[col][sum_check[col] > 0]
                if not valid_vals.empty:
                    observed_min_diff = min(observed_min_diff, valid_vals.min())
        
        if observed_min_diff >= 0.09:
            dynamic_floor = 0.1
        else:
            dynamic_floor = 10 ** np.floor(np.log10(observed_min_diff))
            dynamic_floor = max(dynamic_floor, 1e-6)
        # ------------------------------------------------
        
        summary = model_df.groupby(["Layer", "Status"]).mean(numeric_only=True).reset_index()
        accuracy_summary = model_df.groupby("Layer")["Layer_Correct"].mean().reset_index()
        
        all_layers = sorted(model_df["Layer"].unique())
        all_statuses = ["Correct", "Incorrect"]
        mux = pd.MultiIndex.from_product([all_layers, all_statuses], names=["Layer", "Status"])
        summary = summary.set_index(["Layer", "Status"]).reindex(mux).reset_index()
        
        summary["Prob_Diff"] = summary["Prob_Diff"].clip(lower=dynamic_floor)
        summary["Chosen_Gold_Diff"] = summary["Chosen_Gold_Diff"].clip(lower=dynamic_floor)
        
        # Linear background accuracy execution bars
        ax_bars = ax_margin.twinx()
        ax_bars.bar(
            accuracy_summary["Layer"], accuracy_summary["Layer_Correct"], 
            color="#e3e3e3", width=0.55, edgecolor="#cccccc", alpha=0.4, zorder=1
        )
        ax_bars.set_ylim(-0.05, 1.05)
        ax_bars.set_ylabel("Layer Accuracy (Linear Scale)", fontsize=12, color="#555555", labelpad=10)
        
        # Plot Logarithmic Probability Curves
        ax_margin.set_yscale('log')
        
        for status, color in colors.items():
            status_df = summary[summary["Status"] == status].dropna(subset=["Prob_Diff"])
            if status_df.empty:
                continue
            ax_margin.plot(
                status_df["Layer"], status_df["Prob_Diff"], 
                color=color, linestyle="-", linewidth=3, 
                marker="o", markersize=7, zorder=3
            )
        
        # Overlay Specific Chosen vs Gold Discrepancy Curve
        incorrect_summary = summary[summary["Status"] == "Incorrect"].dropna(subset=["Chosen_Gold_Diff"])
        if not incorrect_summary.empty:
            ax_margin.plot(
                incorrect_summary["Layer"], incorrect_summary["Chosen_Gold_Diff"],
                color="#111111", linestyle="-.", linewidth=2.5, marker="X", markersize=7,
                label="Incorrect Path: Chosen vs Gold Discrepancy", zorder=4
            )
            
        ax_margin.set_xlim(min(all_layers) - 0.6, max(all_layers) + 0.6)
        ax_margin.set_xticks(all_layers)
        ax_margin.set_xticklabels([str(lyr) for lyr in all_layers], fontsize=11)
        
        ax_margin.set_ylim(bottom=dynamic_floor, top=1.05)
        ax_margin.set_xlabel(f"Model Layers", fontweight='bold', fontsize=12)
        ax_margin.set_ylabel(f"Mean Probability Difference (Log Scale)", fontsize=12, color='#333333')
            
        ax_margin.set_title(f"{model}", fontsize=15, fontweight='bold', pad=14)
        ax_margin.grid(True, linestyle=':', alpha=0.5, color='#999999', zorder=0)

    # ---- Outside Legend Construction: Repositioned to the bottom center ----
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color="#e3e3e3", ec="#cccccc", alpha=0.5, label="Layer Accuracy (Linear Scale)"),
        Line2D([0], [0], color=colors["Correct"], lw=3, linestyle="-", marker="o", label="Correct Path: Top-1 vs Top-2 Margin"),
        Line2D([0], [0], color=colors["Incorrect"], lw=3, linestyle="-", marker="o", label="Incorrect Path: Top-1 vs Top-2 Margin"),
        Line2D([0], [0], color="#111111", lw=2.5, linestyle="-.", marker="X", label="Incorrect Path: Chosen vs Gold Discrepancy (P_Top1 - P_Gold)")
    ]
    
    fig.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.08),  # Anchored below the plots
        ncol=2,                      # Arranged in 2 neat rows/columns for a horizontal block layout
        frameon=True, facecolor='white', edgecolor='#cccccc',
        title="Prediction Competition Layers", title_fontsize=12, fontsize=11
    )
    
    plt.suptitle(
        "Probability Difference Distribution Across Layers", 
        y=0.98, fontsize=16, fontweight='bold'
    )
    
    out_path = os.path.join(OUTPUT_DIR, "Symbolic_Equation_qwen_vs_llama_margins.png")
    # Added bbox_extra_artists to prevent the legend cut-off at the bottom edge
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Cross-model Symbolic Equation grid matrix with bottom legend generated: {out_path}")

if __name__ == "__main__":
    print("Parsing inference metrics directory data...")
    margin_df = parse_margin_metrics("inference_eos_uni")
    
    if not margin_df.empty:
        print("Generating decoupled cross-model comparison for Symbolic Equation...")
        plot_cross_model_worded_margin_grid(margin_df, num_last_layers=15)
        print("Success!")
    else:
        print("❌ Error: Could not parse any trace data matches inside the target directories.")