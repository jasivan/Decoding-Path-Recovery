# import os
# import json
# import glob
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set clean scientific canvas layout defaults
# sns.set_theme(style="whitegrid", context="talk")
# plt.rcParams.update({
#     'figure.max_open_warning': 100, 
#     'font.size': 13,
#     'axes.edgecolor': '#cccccc',
#     'axes.linewidth': 1.5
# })

# OUTPUT_DIR = "inference_eos_uni"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def parse_multi_model_normalized_layers(data_dir="."):
#     """Sweeps all inference JSONs, extracts step-0 accuracy per layer,

#     and normalizes layer depth into relative percentages (0-100%).
#     """
#     records = []
#     file_pattern = os.path.join(data_dir, "inference_data_*.json")
#     files = glob.glob(file_pattern)
    
#     raw_entries = []
#     for file_path in files:
#         file_name = os.path.basename(file_path)
#         model_name = file_name.replace("inference_data_", "").split("_")[0]
#         if not model_name or model_name.endswith(".json"):
#             model_name = "Unknown_Model"
            
#         format_type = "Worded_Question" if "question_fixed.json" in file_name else "Symbolic_Equation"
            
#         with open(file_path, 'r') as f:
#             try:
#                 data = json.load(f)
#                 if isinstance(data, dict): data = [data]
#             except json.JSONDecodeError:
#                 continue

#             for entry in data:
#                 layer_traces = entry.get("layer_traces", {})
#                 valid_layers_per_step = layer_traces.get("ml_valid_layers_per_step", [])
#                 step_0_valid_layers = valid_layers_per_step[0] if valid_layers_per_step else []
                
#                 fine_grain = layer_traces.get("ml_layer_predictions_fine_grain", [])
#                 if not fine_grain:
#                     continue
                    
#                 first_step_layers = fine_grain[0]
#                 for layer_key in first_step_layers.keys():
#                     try:
#                         layer_idx = int(layer_key.replace("layer_", ""))
#                         is_layer_correct = 1 if layer_idx in step_0_valid_layers else 0
                        
#                         raw_entries.append({
#                             "Model": model_name,
#                             "Format": format_type,
#                             "Layer": layer_idx,
#                             "Layer_Correct": is_layer_correct
#                         })
#                     except (ValueError, TypeError):
#                         continue

#     if not raw_entries:
#         return pd.DataFrame()
        
#     df_raw = pd.DataFrame(raw_entries)
    
#     normalized_records = []
#     for model in df_raw["Model"].unique():
#         model_subset = df_raw[df_raw["Model"] == model].copy()
#         max_layer = model_subset["Layer"].max()
        
#         if max_layer == 0: 
#             max_layer = 1
            
#         model_subset["Layer_Depth_Pct"] = (model_subset["Layer"] / max_layer) * 100
#         normalized_records.append(model_subset)
        
#     return pd.concat(normalized_records, ignore_index=True)

# def plot_cross_model_alignment_late_layers(df, min_pct_cutoff=50.0):
#     """Plots comparative trajectories with user-enforced structural model family ordering."""
#     if df.empty:
#         print("No evaluation matrices found across logs.")
#         return

#     formats = ["Symbolic_Equation", "Worded_Question"]
#     format_titles = ["Symbolic Equation Context", "Worded Question Context"]
    
#     # ----------------------------------------------------
#     # SETUP EXPLICIT USER-REQUESTED MODEL ORDER & STYLING
#     # ----------------------------------------------------
#     requested_order = [
#         "Llama-3.2-1B",
#         "Llama-3.2-3B",
#         "Llama-3.1-8B",
#         "Qwen3-0.6B",
#         "Qwen3-1.7B",
#         "Qwen3-4B",
#         "Qwen3-8B",
#         "Qwen3-14B"
#     ]
    
#     # Identify discovered models and align them using your targeted configuration sequence
#     discovered_models = df["Model"].unique()
#     ordered_models = [m for m in requested_order if m in discovered_models]
    
#     # Capture any anomalous models not explicitly stated in your requested order sequence
#     for m in discovered_models:
#         if m not in ordered_models:
#             ordered_models.append(m)
            
#     # Generate distinct colors matched specifically against the custom order mapping
#     color_palette = sns.color_palette("tab10", len(ordered_models))
#     model_colors = dict(zip(ordered_models, color_palette))
    
#     # Assign specific linestyles based on target architectural families
#     model_styles = {}
#     for model in ordered_models:
#         if "llama" in model.lower():
#             model_styles[model] = "--"  # Dashed for Llama
#         elif "qwen" in model.lower():
#             model_styles[model] = "-"   # Solid for Qwen
#         else:
#             model_styles[model] = ":"   # Dotted fallback
            
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
#     plt.subplots_adjust(wspace=0.15)
    
#     for idx, fmt in enumerate(formats):
#         ax = axes[idx]
#         slice_df = df[df["Format"] == fmt].copy()
        
#         if slice_df.empty:
#             ax.text(0.5, 0.5, "Empty Format Subset", ha='center', va='center')
#             continue
            
#         slice_df["Layer_Depth_Pct_Group"] = slice_df["Layer_Depth_Pct"].round(0)
#         slice_df = slice_df[slice_df["Layer_Depth_Pct_Group"] >= min_pct_cutoff]
        
#         summary = slice_df.groupby(["Model", "Layer_Depth_Pct_Group"])["Layer_Correct"].mean().reset_index()
#         summary["Accuracy_Pct"] = summary["Layer_Correct"] * 100
        
#         # Plot each model individually following our explicit ordered list structure
#         for model in ordered_models:
#             model_summary = summary[summary["Model"] == model]
#             if model_summary.empty:
#                 continue
                
#             ax.plot(
#                 model_summary["Layer_Depth_Pct_Group"],
#                 model_summary["Accuracy_Pct"],
#                 color=model_colors[model],
#                 linestyle=model_styles[model],
#                 linewidth=3,
#                 marker="o",
#                 markersize=6,
#                 alpha=0.9,
#                 label=model
#             )
        
#         ax.set_title(format_titles[idx], fontsize=15, fontweight='bold', pad=14)
#         ax.set_xlabel("Relative Network Depth (% of Total Layers)", fontsize=13)
#         ax.set_xlim(min_pct_cutoff - 2, 102)
#         ax.set_xticks(np.arange(int(min_pct_cutoff), 101, 10))
#         ax.set_ylim(-5, 105)
        
#         if idx == 0:
#             ax.set_ylabel("Internal Layer Accuracy Rate (%)", fontsize=13)
            
#         ax.grid(True, which="both", linestyle='--', linewidth=0.7, color='#e0e0e0')

#     # ---- Build Enforced Custom Master Legend Matrix ----
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color=model_colors[m], linestyle=model_styles[m], lw=3, marker="o", label=m)
#         for m in ordered_models
#     ]
    
#     fig.legend(
#         handles=legend_elements, 
#         loc='center left', 
#         bbox_to_anchor=(0.92, 0.5),
#         frameon=True,
#         facecolor='white',
#         edgecolor='#cccccc',
#         fontsize=12,
#         title="Architectures (Ordered)", 
#         title_fontsize=13
#     )
    
#     plt.suptitle(f"Cross-Model Performance Alignment by Layer", y=0.98, fontsize=16, fontweight='bold')
    
#     out_path = os.path.join(OUTPUT_DIR, f"cross_model_depth_accuracy_from_{int(min_pct_cutoff)}pct.png")
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f" -> Styled cross-model analysis script completed. Output generated at: {out_path}")

# if __name__ == "__main__":
#     print("Parsing model logs and organizing layer matrices...")
#     normalized_df = parse_multi_model_normalized_layers("inference_eos_uni")
#     plot_cross_model_alignment_late_layers(normalized_df, min_pct_cutoff=50.0)

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
    'font.size': 13,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.5
})

OUTPUT_DIR = "inference_eos_uni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_multi_model_normalized_layers(data_dir="."):
    """Sweeps all inference JSONs, extracts step-0 accuracy per layer,
    and normalizes layer depth into relative percentages (0-100%).
    """
    records = []
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    raw_entries = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0].strip('-Instruct')
        if not model_name or model_name.endswith(".json"):
            model_name = "Unknown_Model"
            
        format_type = "Worded_Question" if "question_fixed.json" in file_name else "Symbolic_Equation"
            
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except json.JSONDecodeError:
                continue

            for entry in data:
                layer_traces = entry.get("layer_traces", {})
                valid_layers_per_step = layer_traces.get("ml_valid_layers_per_step", [])
                step_0_valid_layers = valid_layers_per_step[0] if valid_layers_per_step else []
                
                fine_grain = layer_traces.get("ml_layer_predictions_fine_grain", [])
                if not fine_grain:
                    continue
                    
                first_step_layers = fine_grain[0]
                for layer_key in first_step_layers.keys():
                    try:
                        layer_idx = int(layer_key.replace("layer_", ""))
                        is_layer_correct = 1 if layer_idx in step_0_valid_layers else 0
                        
                        raw_entries.append({
                            "Model": model_name,
                            "Format": format_type,
                            "Layer": layer_idx,
                            "Layer_Correct": is_layer_correct
                        })
                    except (ValueError, TypeError):
                        continue

    if not raw_entries:
        return pd.DataFrame()
        
    df_raw = pd.DataFrame(raw_entries)
    
    normalized_records = []
    for model in df_raw["Model"].unique():
        model_subset = df_raw[df_raw["Model"] == model].copy()
        max_layer = model_subset["Layer"].max()
        
        if max_layer == 0: 
            max_layer = 1
            
        model_subset["Layer_Depth_Pct"] = (model_subset["Layer"] / max_layer) * 100
        normalized_records.append(model_subset)
        
    return pd.concat(normalized_records, ignore_index=True)

def plot_cross_model_alignment_late_layers(df, min_pct_cutoff=50.0):
    """Plots comparative trajectories exclusively for the Symbolic Equation Context."""
    if df.empty:
        print("No evaluation matrices found across logs.")
        return

    # Filter strictly for Symbolic_Equation immediately
    slice_df = df[df["Format"] == "Symbolic_Equation"].copy()
    if slice_df.empty:
        print("No Symbolic Equation entries found in the dataset.")
        return
    
    # ----------------------------------------------------
    # SETUP EXPLICIT USER-REQUESTED MODEL ORDER & STYLING
    # ----------------------------------------------------
    requested_order = [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        "Llama-3.1-8B",
    ]
    
    discovered_models = slice_df["Model"].unique()
    ordered_models = [m for m in requested_order if m in discovered_models]
    
    for m in discovered_models:
        if m not in ordered_models:
            ordered_models.append(m)
            
    color_palette = sns.color_palette("tab10", len(ordered_models))
    model_colors = dict(zip(ordered_models, color_palette))
    
    model_styles = {}
    for model in ordered_models:
        if "llama" in model.lower():
            model_styles[model] = "--"  # Dashed for Llama
        elif "qwen" in model.lower():
            model_styles[model] = "-"   # Solid for Qwen
        else:
            model_styles[model] = ":"   # Dotted fallback
            
    # Set up single plot axes (11x8 to preserve readable aspect ratios)
    fig, ax = plt.subplots(figsize=(11, 8))
    
    slice_df["Layer_Depth_Pct_Group"] = slice_df["Layer_Depth_Pct"].round(0)
    slice_df = slice_df[slice_df["Layer_Depth_Pct_Group"] >= min_pct_cutoff]
    
    summary = slice_df.groupby(["Model", "Layer_Depth_Pct_Group"])["Layer_Correct"].mean().reset_index()
    summary["Accuracy_Pct"] = summary["Layer_Correct"] * 100
    
    # Plot each model following explicit list structure sequence
    for model in ordered_models:
        model_summary = summary[summary["Model"] == model]
        if model_summary.empty:
            continue
            
        ax.plot(
            model_summary["Layer_Depth_Pct_Group"],
            model_summary["Accuracy_Pct"],
            color=model_colors[model],
            linestyle=model_styles[model],
            linewidth=3,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=model
        )
    
    ax.set_title("Symbolic Equation", fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel("Normalised Model Depth (% of Total Layers)", fontsize=13)
    ax.set_xlim(min_pct_cutoff - 2, 102)
    ax.set_xticks(np.arange(int(min_pct_cutoff), 101, 10))
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Internal Layer Accuracy Rate (%)", fontsize=13)
        
    ax.grid(True, which="both", linestyle='--', linewidth=0.7, color='#e0e0e0')

    # ---- Build Enforced Custom Master Legend Matrix ----
    legend_elements = [
        Line2D([0], [0], color=model_colors[m], linestyle=model_styles[m], lw=3, marker="o", label=m)
        for m in ordered_models
    ]
    
    ax.legend(
        handles=legend_elements, 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        fontsize=12,
        title="Models", 
        title_fontsize=13
    )
    
    plt.suptitle("Cross-Model Performance by Layer", y=0.98, fontsize=16, fontweight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, f"cross_model_depth_accuracy_symbolic_{int(min_pct_cutoff)}pct.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Styled cross-model analysis script completed. Output generated at: {out_path}")

if __name__ == "__main__":
    print("Parsing model logs and organizing layer matrices...")
    normalized_df = parse_multi_model_normalized_layers("inference_eos_uni")
    plot_cross_model_alignment_late_layers(normalized_df, min_pct_cutoff=50.0)