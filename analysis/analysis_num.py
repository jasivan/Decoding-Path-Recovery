# import os
# import json
# import glob
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set clean scientific canvas layout defaults
# sns.set_theme(style="white", context="talk")
# plt.rcParams.update({'figure.max_open_warning': 100, 'font.size': 11})

# OUTPUT_DIR = "inference_eos_uni"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def clean_subgroup_string(raw_val):
#     """Normalizes variation in file logs to ensure precise alignment."""
#     s = str(raw_val).lower().replace(" ", "").strip()
#     if "2digit" in s: return "2 digit"
#     if "3digit" in s: return "3 digit"
#     if "4digit" in s and "+" not in s: return "4 digit"
#     if "4+digit" in s or "4digit+" in s: return "4+ digit"
#     if "1dp" in s: return "1dp"
#     if "2dp" in s: return "2dp"
#     return str(raw_val).strip()

# def parse_fine_grain_accuracy_traces(data_dir="inference_eos_uni"):
#     """Sweeps json logs and computes both Sequence-Wise and Token-Wise correctness

#     at the operational step level for comparative plotting.
#     """
#     flat_records = []
#     file_pattern = os.path.join(data_dir, "inference_data_*.json")
#     files = glob.glob(file_pattern)
    
#     for file_path in files:
#         file_name = os.path.basename(file_path)
#         model_name = file_name.replace("inference_data_", "").split("_")[0]
#         if not model_name or model_name.endswith(".json"):
#             model_name = "Qwen3-14B"
            
#         if file_name.endswith("_question_fixed.json"):
#             format_type = "Worded_Question"
#         elif file_name.endswith("hard.json"):
#             format_type = "Symbolic_Equation"
#         else:
#             continue
            
#         with open(file_path, 'r') as f:
#             try:
#                 data = json.load(f)
#                 if isinstance(data, dict): data = [data]
#             except json.JSONDecodeError:
#                 continue

#             for entry in data:
#                 subgroup_label = clean_subgroup_string(entry.get("subgroup", "Unknown"))
#                 operation = entry.get("operation", "Unknown")
                
#                 metrics = entry.get("metrics", {})
#                 seq_correct = bool(metrics.get("ml_str_match", False))
#                 seq_label = "Seq_Correct" if seq_correct else "Seq_Incorrect"
                
#                 layer_traces = entry.get("layer_traces", {})
#                 prob_mass_steps = layer_traces.get("ml_numeric_prob_mass_per_step", [])
                
#                 gen_tokens = layer_traces.get("generated_tokens_per_step", [])
#                 gold_tokens = layer_traces.get("gold_tokens_per_step", [])
#                 layer_preds_per_step = layer_traces.get("layer_predictions_per_step", [])

#                 for step_idx, step_dict in enumerate(prob_mass_steps):
#                     gen_stage = "First Token (Step 0)" if step_idx == 0 else "Subsequent Tokens (Step 1+)"
                    
#                     # Compute Token-Wise (TK) Correctness
#                     tk_correct = False
#                     if step_idx < len(layer_preds_per_step) and step_idx < len(gold_tokens):
#                         target_token = str(gold_tokens[step_idx]).strip()
#                         step_layer_preds = layer_preds_per_step[step_idx]
#                         if isinstance(step_layer_preds, dict):
#                             tk_correct = any(str(tok).strip() == target_token for tok in step_layer_preds.values())
#                         elif isinstance(step_layer_preds, list):
#                             tk_correct = any(str(tok).strip() == target_token for tok in step_layer_preds)

#                     if not tk_correct and step_idx < len(gen_tokens) and step_idx < len(gold_tokens):
#                         tk_correct = (str(gen_tokens[step_idx]).strip() == str(gold_tokens[step_idx]).strip())
                    
#                     if not gen_tokens and not layer_preds_per_step:
#                         tk_correct = seq_correct

#                     tk_label = "TK_Correct" if tk_correct else "TK_Incorrect"
                    
#                     for k, v in step_dict.items():
#                         try:
#                             layer_idx = int(''.join(filter(str.isdigit, k)))
#                             flat_records.append({
#                                 "Model": model_name,
#                                 "Format": format_type,
#                                 "Seq_Accuracy": seq_label,
#                                 "TK_Accuracy": tk_label,
#                                 "Generation_Stage": gen_stage,
#                                 "Subgroup": subgroup_label,
#                                 "Operation": operation,
#                                 "Layer": layer_idx,
#                                 "Cum_Prob": float(v)
#                             })
#                         except ValueError:
#                             continue

#     return pd.DataFrame(flat_records)

# def generate_accuracy_split_dashboard(df, model_name, format_type, accuracy_type, all_ops):
#     """Generates the primary heatmaps matching the step-aligned evaluation constraints."""
#     if accuracy_type == "All_Combined":
#         subset_df = df[(df["Model"] == model_name) & (df["Format"] == format_type)].copy()
#     else:
#         tk_track = "TK_Correct" if "Correct" in accuracy_type else "TK_Incorrect"
#         subset_df = df[(df["Model"] == model_name) & (df["Format"] == format_type) & (df["TK_Accuracy"] == tk_track)].copy()
        
#     if subset_df.empty:
#         return

#     total_layers = df[(df["Model"] == model_name)]["Layer"].max()
#     midpoint_layer = int(total_layers / 2)
#     subset_df = subset_df[subset_df["Layer"] >= midpoint_layer]

#     fig = plt.figure(figsize=(22, 24))
#     gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
#     stages = ["First Token (Step 0)", "Subsequent Tokens (Step 1+)"]
#     heatmap_kwargs = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
#     all_layers = list(range(midpoint_layer, total_layers + 1))
#     preferred_order = ["2 digit", "3 digit", "4 digit", "4+ digit", "1dp", "2dp"]

#     for col_idx, stage in enumerate(stages):
#         stage_df = subset_df[subset_df["Generation_Stage"] == stage]
#         show_cbar = (col_idx == 1)
#         cbar_args = {'label': 'Mean Prob Mass'} if show_cbar else None

#         # ROW 1: Operations
#         ax_op = fig.add_subplot(gs[0, col_idx])
#         if not stage_df.empty:
#             op_pivot = stage_df.groupby(["Operation", "Layer"])["Cum_Prob"].mean().unstack(level="Layer")
#             op_pivot = op_pivot.reindex(index=all_ops, columns=all_layers, fill_value=0.0)
#         else:
#             op_pivot = pd.DataFrame(0.0, index=all_ops, columns=all_layers)
#         sns.heatmap(op_pivot, ax=ax_op, cbar=show_cbar, cbar_kws=cbar_args, **heatmap_kwargs)
#         ax_op.set_title(f"{stage}\nStratified Matrix: Mathematical Operations", fontsize=12, fontweight='bold')
#         ax_op.set_ylabel("Operations")
#         ax_op.set_xlabel("").set_visible(False)

#         # ROW 2: Subgroups
#         ax_sub = fig.add_subplot(gs[1, col_idx])
#         if not stage_df.empty:
#             sub_pivot = stage_df.groupby(["Subgroup", "Layer"])["Cum_Prob"].mean().unstack(level="Layer")
#             for missing_group in preferred_order:
#                 if missing_group not in sub_pivot.index: sub_pivot.loc[missing_group] = 0.0
#             sub_pivot = sub_pivot.loc[preferred_order, all_layers].fillna(0.0)
#         else:
#             sub_pivot = pd.DataFrame(0.0, index=preferred_order, columns=all_layers)
#         sns.heatmap(sub_pivot, ax=ax_sub, cbar=show_cbar, cbar_kws=cbar_args, **heatmap_kwargs)
#         ax_sub.set_title("Stratified Matrix: Subgroup Context Scales", fontsize=12, fontweight='bold')
#         ax_sub.set_ylabel("Problem Types / Scales")
#         ax_sub.set_xlabel("").set_visible(False)

#         # ROW 3: Aggregate Summary
#         ax_top = fig.add_subplot(gs[2, col_idx])
#         if not stage_df.empty:
#             global_pivot = stage_df.groupby("Layer")["Cum_Prob"].mean().to_frame().T
#             global_pivot.index = ["All Combined"]
#             global_pivot = global_pivot.reindex(columns=all_layers, fill_value=0.0)
#         else:
#             global_pivot = pd.DataFrame(0.0, index=["All Combined"], columns=all_layers)
#         sns.heatmap(global_pivot, ax=ax_top, cbar=show_cbar, cbar_kws=cbar_args, **heatmap_kwargs)
#         ax_top.set_title("Stratified Matrix: Global Aggregate Summary Baseline", fontsize=12, fontweight='bold')
#         ax_top.set_ylabel("Overall")
#         ax_top.set_xlabel(f"Transformer Layer Index (Truncated: Layers {midpoint_layer} to {total_layers})")

#     plt.suptitle(f"Model Diagnostic Profile: {model_name} [{format_type.replace('_', ' ')}] ({accuracy_type})\nStep-Level Shared Evaluation (Final $50\\%$ of Network Layers)", y=0.99, fontsize=15, fontweight='bold')
#     out_path = os.path.join(OUTPUT_DIR, f"{model_name}_{format_type}_{accuracy_type}_step_aligned_heatmaps.png")
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()

# def generate_comparative_line_plots(df):
#     """Generates two line graphs tracking probability trajectories across the final 50% of layers.

#     Enforces exact styling properties: Green=Correct, Red=Incorrect, Black=All, 
#     Dashed=Symbolic, Solid=Worded.
#     """
#     total_layers = df["Layer"].max()
#     midpoint_layer = int(total_layers / 2)
#     line_df = df[df["Layer"] >= midpoint_layer].copy()
    
#     if line_df.empty:
#         return

#     def plot_format_tracks(data, accuracy_col, accuracy_val, label_suffix, color_hex):
#         """Helper to explicitly render lines with fixed colors and custom linestyles."""
#         # Worded Track (Solid Line Style)
#         w_df = data[(data[accuracy_col] == accuracy_val) & (data["Format"] == "Worded_Question")]
#         if not w_df.empty:
#             w_mean = w_df.groupby("Layer")["Cum_Prob"].mean().reset_index()
#             plt.plot(w_mean["Layer"], w_mean["Cum_Prob"], color=color_hex, linestyle="-", linewidth=3.5, label=f"Worded ({label_suffix})")
            
#         # Symbolic Track (Dashed Line Style)
#         s_df = data[(data[accuracy_col] == accuracy_val) & (data["Format"] == "Symbolic_Equation")]
#         if not s_df.empty:
#             s_mean = s_df.groupby("Layer")["Cum_Prob"].mean().reset_index()
#             plt.plot(s_mean["Layer"], s_mean["Cum_Prob"], color=color_hex, linestyle="--", linewidth=3.5, label=f"Symbolic ({label_suffix})")

#     def plot_all_baseline_tracks(data, color_hex):
#         """Helper to render the baseline summary averages across all instances."""
#         # All Worded Track (Solid Black Line)
#         w_df = data[data["Format"] == "Worded_Question"]
#         if not w_df.empty:
#             w_mean = w_df.groupby("Layer")["Cum_Prob"].mean().reset_index()
#             plt.plot(w_mean["Layer"], w_mean["Cum_Prob"], color=color_hex, linestyle="-", linewidth=3.5, label="Worded (All)")
            
#         # All Symbolic Track (Dashed Black Line)
#         s_df = data[data["Format"] == "Symbolic_Equation"]
#         if not s_df.empty:
#             s_mean = s_df.groupby("Layer")["Cum_Prob"].mean().reset_index()
#             plt.plot(s_mean["Layer"], s_mean["Cum_Prob"], color=color_hex, linestyle="--", linewidth=3.5, label="Symbolic (All)")

#     # ---------------------------------------------------------
#     # GRAPH A: Sequence-Wise (ML-Wise Outcome Match)
#     # ---------------------------------------------------------
#     plt.figure(figsize=(13, 8))
#     plot_format_tracks(line_df, "Seq_Accuracy", "Seq_Correct", "Correct", "green")
#     plot_format_tracks(line_df, "Seq_Accuracy", "Seq_Incorrect", "Incorrect", "red")
#     plot_all_baseline_tracks(line_df, "black")
    
#     plt.title("Sequence-Wise (ML-Wise) Probability Mass Trajectory\nFormat Comparison Across Late-Stage Network Layers", fontsize=13, fontweight='bold')
#     plt.xlabel(f"Transformer Layer Index (Late Stage: Layers {midpoint_layer} to {total_layers})")
#     plt.ylabel("Mean Cumulative Probability Mass")
#     plt.ylim(-0.05, 1.05)
#     plt.grid(True, linestyle=":", alpha=0.5)
#     plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
#     plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_sequence_wise_lineplot.png"), dpi=300, bbox_inches='tight')
#     plt.close()

#     # ---------------------------------------------------------
#     # GRAPH B: Token-Wise (TK-Wise Step Token Match)
#     # ---------------------------------------------------------
#     plt.figure(figsize=(13, 8))
#     plot_format_tracks(line_df, "TK_Accuracy", "TK_Correct", "Correct", "green")
#     plot_format_tracks(line_df, "TK_Accuracy", "TK_Incorrect", "Incorrect", "red")
#     plot_all_baseline_tracks(line_df, "black")
    
#     plt.title("Token-Wise (TK-Wise) Probability Mass Trajectory\nFormat Comparison Across Late-Stage Network Layers", fontsize=13, fontweight='bold')
#     plt.xlabel(f"Transformer Layer Index (Late Stage: Layers {midpoint_layer} to {total_layers})")
#     plt.ylabel("Mean Cumulative Probability Mass")
#     plt.ylim(-0.05, 1.05)
#     plt.grid(True, linestyle=":", alpha=0.5)
#     plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
#     plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_token_wise_lineplot.png"), dpi=300, bbox_inches='tight')
#     plt.close()


# if __name__ == "__main__":
#     print("Parsing logs and running unified matrix/line plot metrics execution...")
#     master_df = parse_fine_grain_accuracy_traces("inference_eos_uni")
    
#     if not master_df.empty:
#         print(" -> Exporting custom-mapped trajectory line plots...")
#         generate_comparative_line_plots(master_df)
        
#         models = master_df["Model"].unique()
#         formats = master_df["Format"].unique()
#         evaluation_tracks = ["All_Combined", "ML_Correct", "ML_Incorrect"]
#         global_ops = sorted(master_df["Operation"].unique())
        
#         for m in models:
#             for f in formats:
#                 for track in evaluation_tracks:
#                     print(f" -> Exporting heatmaps for Model: {m} | Format: {f} | Split: {track}...")
#                     generate_accuracy_split_dashboard(master_df, m, f, track, global_ops)
                    
#         print(f"\nSuccess! All outputs rendered and exported to: ./{OUTPUT_DIR}/")
#     else:
#         print("Data parsing execution failed: Ensure trace arrays populate inner folders.")

import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_specific_comparison_data(data_dir="inference_eos_uni"):
    prob_records = []
    layer_acc_records = []
    
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    target_models = ["Qwen3-8B", "Llama-3.1-8B-Instruct"]

    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0]
        
        # Enforce target model filters
        if model_name not in target_models:
            continue
            
        # Target Equations / Symbolic context only
        if "question_fixed.json" in file_name or not file_name.endswith("hard.json"):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except Exception:
                continue

            for entry in data:
                layer_traces = entry.get("layer_traces", {})
                valid_layers_per_step = layer_traces.get("ml_valid_layers_per_step", [])
                prob_mass_steps = layer_traces.get("ml_numeric_prob_mass_per_step", [])
                gen_tokens = layer_traces.get("generated_tokens_per_step", [])
                gold_tokens = layer_traces.get("gold_tokens_per_step", [])
                layer_preds_per_step = layer_traces.get("layer_predictions_per_step", [])
                metrics = entry.get("metrics", {})
                seq_correct = bool(metrics.get("ml_str_match", False))

                if not valid_layers_per_step or not prob_mass_steps:
                    continue

                for step_idx, step_dict in enumerate(prob_mass_steps):
                    if step_idx >= len(valid_layers_per_step):
                        continue
                        
                    # Split condition type
                    cond_label = "First_Token" if step_idx == 0 else "Subsequent_Tokens"
                    
                    valid_set = set(int(x) for x in valid_layers_per_step[step_idx])

                    # Token correctness evaluation for line hue grouping
                    tk_correct = False
                    if step_idx < len(layer_preds_per_step) and step_idx < len(gold_tokens):
                        target_token = str(gold_tokens[step_idx]).strip()
                        step_layer_preds = layer_preds_per_step[step_idx]
                        if isinstance(step_layer_preds, dict):
                            tk_correct = any(str(tok).strip() == target_token for tok in step_layer_preds.values())
                        elif isinstance(step_layer_preds, list):
                            tk_correct = any(str(tok).strip() == target_token for tok in step_layer_preds)

                    if not tk_correct and step_idx < len(gen_tokens) and step_idx < len(gold_tokens):
                        tk_correct = (str(gen_tokens[step_idx]).strip() == str(gold_tokens[step_idx]).strip())
                    
                    if not gen_tokens and not layer_preds_per_step:
                        tk_correct = seq_correct

                    tk_label = "TK_Correct" if tk_correct else "TK_Incorrect"

                    for k, v in step_dict.items():
                        try:
                            layer_idx = int(''.join(filter(str.isdigit, k)))
                            is_layer_correct = 1 if layer_idx in valid_set else 0
                            
                            layer_acc_records.append({
                                "Model": model_name, "Condition": cond_label,
                                "Layer": layer_idx, "Is_Correct": is_layer_correct
                            })
                            
                            prob_records.append({
                                "Model": model_name, "Condition": cond_label,
                                "TK_Accuracy": tk_label, "Layer": layer_idx,
                                "Cum_Prob": float(v)
                            })
                        except ValueError:
                            continue

    if not prob_records:
        print("⚠️ No equation matching records found for Qwen3-8B and Llama-3.1-8B.")
        return pd.DataFrame(), pd.DataFrame()

    df_prob = pd.DataFrame(prob_records)
    df_acc = pd.DataFrame(layer_acc_records)

    # Normalize depth array to relative percentages
    for df in [df_prob, df_acc]:
        normalized_dfs = []
        for model in df["Model"].unique():
            sub = df[df["Model"] == model].copy()
            max_lyr = sub["Layer"].max()
            sub["Layer_Depth_Pct"] = ((sub["Layer"] / (max_lyr if max_lyr > 0 else 1)) * 100).round().astype(int)
            normalized_dfs.append(sub)
        if df is df_prob:
            df_prob = pd.concat(normalized_dfs, ignore_index=True)
        else:
            df_acc = pd.concat(normalized_dfs, ignore_index=True)

    # Filter to look strictly from 50% relative depth onwards
    df_prob = df_prob[df_prob["Layer_Depth_Pct"] >= 50]
    df_acc = df_acc[df_acc["Layer_Depth_Pct"] >= 50]

    return df_prob, df_acc

def render_focused_comparison_plot(df_prob, df_acc, data_dir="."):
    if df_prob.empty:
        return

    # Strict ordering requirements
    row_models = ["Qwen3-8B", "Llama-3.1-8B-Instruct"]
    col_conditions = ["First_Token", "Subsequent_Tokens"]
    
    # Decouple shared axes to ensure grids render seamlessly across independent dual axes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 11))
    
    # Tightened vertical/horizontal spacing and title clearances
    plt.subplots_adjust(wspace=0.18, hspace=0.18, top=0.90, bottom=0.12)
    
    # Uniform color palette mappings matching the margin plots
    colors = {"TK_Correct": "#2ca02c", "TK_Incorrect": "#d62728"}

    for r_idx, model in enumerate(row_models):
        for c_idx, cond in enumerate(col_conditions):
            ax1 = axes[r_idx, c_idx]
            
            p_slice = df_prob[(df_prob["Model"] == model) & (df_prob["Condition"] == cond)]
            a_slice = df_acc[(df_acc["Model"] == model) & (df_acc["Condition"] == cond)]
            
            if p_slice.empty:
                ax1.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
                continue
                
            depth_bins = sorted(p_slice["Layer_Depth_Pct"].unique())
            acc_grouped = a_slice.groupby("Layer_Depth_Pct")["Is_Correct"].mean().reindex(depth_bins, fill_value=0)
            correct_pcts = (acc_grouped * 100).values

            # ---- 1. BACKGROUND ACCURACY BARS ----
            ax_bars = ax1.twinx()
            ax_bars.bar(
                depth_bins, correct_pcts, 
                color="#e3e3e3", width=1.6, edgecolor="#cccccc", alpha=0.4, zorder=1
            )
            ax_bars.set_ylim(-5, 105)
            
            # Label cleanups for primary right y-axis values (only show on far right column)
            if c_idx == 1:
                ax_bars.set_ylabel("Layer Accuracy (Linear Scale %)", fontsize=11, color="#555555", labelpad=10)
            else:
                ax_bars.set_yticklabels([])

            # ---- 2. FOREGROUND CONTINUOUS PROB LINES ----
            sns.lineplot(
                data=p_slice, x="Layer_Depth_Pct", y="Cum_Prob", hue="TK_Accuracy",
                hue_order=["TK_Correct", "TK_Incorrect"],
                palette=colors, 
                linewidth=3, marker="o", markersize=7, ax=ax1, errorbar=None, legend=False,
                zorder=3
            )
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlim(47, 103)
            
            # Axis labels adjustments per position
            if c_idx == 0:
                ax1.set_ylabel(f"{model}\nMean Numeric Token Probability Mass", fontsize=12, fontweight='bold', color="#333333")
            else:
                ax1.set_ylabel("")
                ax1.set_yticklabels([]) # Clear redundant middle y-ticks

            if r_idx == 1:
                ax1.set_xlabel("Relative Depth (%)", fontweight='bold', fontsize=12)
            else:
                ax1.set_xlabel("")
                ax1.set_xticklabels([]) # Clear redundant upper x-ticks

            # Column headings on the top row panels
            if r_idx == 0:
                title_label = "Initiation Generation (Step 0)" if cond == "First_Token" else "Continuation Generation (Steps 1+)"
                ax1.set_title(title_label, fontsize=14, fontweight='bold', pad=10)

            # ---- 3. SEAMLESS BACKGROUND GRID INTEGRATION ----
            ax1.grid(True, linestyle=':', alpha=0.5, color='#999999', zorder=0)

    # ---- Clean Consolidated Single-Line Bottom Legend Construction ----
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color="#e3e3e3", ec="#cccccc", alpha=0.5, label="Layer Accuracy"),
        plt.Line2D([0], [0], color=colors["TK_Correct"], lw=3, linestyle="-", marker="o", label="Correct Predicted Token"),
        plt.Line2D([0], [0], color=colors["TK_Incorrect"], lw=3, linestyle="-", marker="o", label="Incorrect Predicted Token")
    ]
    
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.03),  # Tucked cleanly beneath bottom axis text
        ncol=3,                      # Forces all three items into a clean single line
        frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11
    )
    
    plt.suptitle("Layer Accuracy for Symbolic Equation across Generation Steps", 
                 fontsize=16, fontweight='bold', y=0.96)

    out_file = os.path.join(data_dir, "qwen_vs_llama_focused_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Matrix plot compiled cleanly with refreshed styles: {out_file}")
    
if __name__ == '__main__':
    # Adjust path if data directory points elsewhere
    dp, da = extract_specific_comparison_data("inference_eos_uni")
    render_focused_comparison_plot(dp, da, "inference_eos_uni")