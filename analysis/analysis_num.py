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