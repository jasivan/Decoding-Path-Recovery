# import os
# import json
# import glob
# from collections import Counter
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set clean scientific canvas defaults
# sns.set_theme(style="whitegrid", context="talk")
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.edgecolor': '#cccccc',
#     'axes.linewidth': 1.2
# })

# OUTPUT_DIR = "inference_eos_uni"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Enforced categorical series sequence layout
# DESIRED_MODEL_ORDER = [
#     "Qwen3-0.6B", 
#     "Qwen3-1.7B", 
#     "Qwen3-4B", 
#     "Qwen3-8B", 
#     "Qwen3-14B",    
#     "llama-3.2-1B", 
#     "llama-3.2-3B",
#     "llama-3.1-8B",
# ]

# def analyze_aggregated_tk_deviations(data_dir="inference_eos_uni"):
#     """Scans JSON logs specifically for the TK configuration, aggregating all
#        sequence positions to see which alternative rank (1 to 4) was chosen
#        whenever Rank 0 was bypassed.
#     """
#     model_counters = {m: Counter() for m in DESIRED_MODEL_ORDER}
#     matched_any_files = False
    
#     file_pattern = os.path.join(data_dir, "inference_data_*.json")
#     files = glob.glob(file_pattern)
    
#     print(f"Scanning directory... Found {len(files)} total matching json files.")
    
#     for file_path in files:
#         file_name = os.path.basename(file_path)
        
#         # Strip structural prefixes out
#         clean_name = file_name.replace("inference_data_", "")
        
#         # --- FIXED: Robust Model Extraction Check ---
#         matched_model = None
#         for target_model in DESIRED_MODEL_ORDER:
#             # Match tokens by locating base variants (e.g., 'llama-3.1-8b' inside 'Llama-3.1-8B-Instruct...')
#             if clean_name.lower().startswith(target_model.lower()):
#                 matched_model = target_model
#                 break
#             # Fallback if names are located elsewhere in custom substrings
#             elif target_model.lower() in clean_name.lower():
#                 matched_model = target_model
#                 break
                
#         if not matched_model:
#             continue
            
#         matched_any_files = True
            
#         with open(file_path, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 if isinstance(data, dict): data = [data]
#             except (json.JSONDecodeError, UnicodeDecodeError):
#                 continue

#             for entry in data:
#                 metrics = entry.get("metrics", {})
#                 tokens_block = entry.get("tokens", {})
                
#                 if bool(metrics.get("tk_str_match", False)):
#                     continue
                    
#                 ranks = tokens_block.get("tk_chosen_vocabulary_ranks", [])
#                 if not ranks:
#                     continue
                
#                 # Check explicitly for Ranks 1 through 4
#                 for r_val in ranks:
#                     if r_val in [1, 2, 3, 4] or str(r_val) in ["1", "2", "3", "4"]:
#                         model_counters[matched_model][f"Rank {int(r_val)}"] += 1
                        
#     if not matched_any_files:
#         print("❌ CRITICAL: No logs mapped successfully. Verify directory naming structure matches pattern profiles.")
                        
#     return model_counters

# def plot_single_aggregated_chart(model_counters):
#     """Plots a single consolidated stacked bar chart comparing all models 
#        side-by-side using the strict categorical ordering array.
#     """
#     records = []
#     for model_name, counter in model_counters.items():
#         total_deviations = sum(counter.values())
#         if total_deviations == 0:
#             print(f"ℹ️ Note: Found files for {model_name}, but it had 0 occurrences of alternative choices.")
#             for r in range(1, 5):
#                 records.append({
#                     "Model": model_name,
#                     "Alternative_Rank": f"Rank {r}",
#                     "Percentage": 0.0
#                 })
#             continue
            
#         for rank_lbl, count in counter.items():
#             records.append({
#                 "Model": model_name,
#                 "Alternative_Rank": rank_lbl,
#                 "Percentage": 100.0 * count / total_deviations
#             })
            
#     df = pd.DataFrame(records)
    
#     # Explicitly enforce a Categorical dtype with the target order
#     df["Model"] = pd.Categorical(df["Model"], categories=DESIRED_MODEL_ORDER, ordered=True)
    
#     rank_order = ["Rank 1", "Rank 2", "Rank 3", "Rank 4"]
#     palette = sns.color_palette("viridis_r", len(rank_order))
#     color_map = {lbl: palette[i] for i, lbl in enumerate(rank_order)}
    
#     fig, ax = plt.subplots(figsize=(11, 8.5))
    
#     # Pivot rows keeping categorical structure safe
#     pivot_df = df.pivot_table(index="Model", columns="Alternative_Rank", values="Percentage", aggfunc="mean", observed=False)
#     pivot_df = pivot_df.reindex(index=DESIRED_MODEL_ORDER, columns=rank_order).fillna(0.0)
    
#     pivot_df.plot(
#         kind="bar", stacked=True, ax=ax,
#         color=[color_map[col] for col in pivot_df.columns],
#         edgecolor="#222222", linewidth=1.3, width=0.45
#     )
    
#     ax.set_title("Aggregated Distribution of Top-5's Alternate Ranks Chosen by TK", 
#                  fontsize=15, fontweight='bold', pad=18)
#     ax.set_xlabel("Models", fontweight='bold', fontsize=13, labelpad=15)
#     ax.set_ylabel("Selected Rank Share (%)", fontweight='bold', fontsize=13)
    
#     # --- SLANTED X-AXIS LABELS ---
#     ax.set_xticklabels(DESIRED_MODEL_ORDER, rotation=30, ha='right', fontweight='bold', fontsize=12)
#     ax.set_ylim(0, 105)
    
#     # Overlay text annotations on elements with non-zero values
#     for rect in ax.patches:
#         h = rect.get_height()
#         if h > 4.5: 
#             ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + h/2., f"{h:.1f}%",
#                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)
            
#     handles = [plt.Rectangle((0,0),1,1, color=color_map[r], ec="#222222") for r in rank_order]
#     ax.legend(handles, rank_order, loc="center left", bbox_to_anchor=(1.02, 0.5),
#               title="TK Alternate Choice\n(Bypassed Rank 0)", title_fontsize=12, fontsize=11,
#               frameon=True, edgecolor="#cccccc")
    
#     plt.tight_layout()
    
#     out_path = os.path.join(OUTPUT_DIR, "tk_fully_aggregated_rank_selections.png")
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f" -> Rigidly ordered aggregated rank chart saved to: {out_path}")

# if __name__ == "__main__":
#     print("Beginning execution of sequence-wide aggregated rank tracking...")
#     aggregated_counters = analyze_aggregated_tk_deviations("inference_eos_uni")
#     plot_single_aggregated_chart(aggregated_counters)

import os
import json
import glob
import pandas as pd

# Define model order to maintain alignment
DESIRED_MODEL_ORDER = [
    "Qwen3-0.6B", 
    "Qwen3-1.7B", 
    "Qwen3-4B", 
    "Qwen3-8B", 
    "Qwen3-14B",    
    "llama-3.2-1B", 
    "llama-3.2-3B",
    "llama-3.1-8B",
]

def calculate_tk_rank_impact(data_dir="inference_eos_uni"):
    file_pattern = os.path.join(data_dir, "inference_data_*_hard.json")
    files = glob.glob(file_pattern)
    
    # Store lists of outcomes per model
    model_stats = {
        m: {"total": 0, "top1": 0, "top2": 0, "top3": 0, "top4": 0, "top5": 0} 
        for m in DESIRED_MODEL_ORDER
    }
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        clean_name = file_name.replace("inference_data_", "")
        
        matched_model = None
        for target_model in DESIRED_MODEL_ORDER:
            if clean_name.lower().startswith(target_model.lower()) or target_model.lower() in clean_name.lower():
                matched_model = target_model
                break
                
        if not matched_model:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): 
                    data = [data]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for entry in data:
                metrics = entry.get("metrics", {})
                tokens_block = entry.get("tokens", {})
                
                base_match = bool(metrics.get("base_str_match", False))
                tk_match = bool(metrics.get("tk_str_match", False))
                ranks = tokens_block.get("tk_chosen_vocabulary_ranks", [])
                
                ranks_clean = []
                for r in ranks:
                    try:
                        ranks_clean.append(int(r))
                    except (ValueError, TypeError):
                        continue
                
                # --- MATHEMATICALLY CORRECT COHORT MATCHING ---
                # Top-1 is exactly baseline success (Rank 0)
                is_top1 = base_match
                
                # Active recovery happens if TK succeeded
                is_recovered = tk_match
                
                # Evaluate recovery boundaries explicitly relative to Zero-Indexed Ranks
                is_top2 = is_top1 or (is_recovered and any(r == 1 for r in ranks_clean))
                is_top3 = is_top1 or (is_recovered and any(r in [1, 2] for r in ranks_clean))
                is_top4 = is_top1 or (is_recovered and any(r in [1, 2, 3] for r in ranks_clean))
                is_top5 = is_top1 or (is_recovered and any(r in [1, 2, 3, 4] for r in ranks_clean))
                
                model_stats[matched_model]["total"] += 1
                if is_top1: model_stats[matched_model]["top1"] += 1
                if is_top2: model_stats[matched_model]["top2"] += 1
                if is_top3: model_stats[matched_model]["top3"] += 1
                if is_top4: model_stats[matched_model]["top4"] += 1
                if is_top5: model_stats[matched_model]["top5"] += 1

    # Print out comprehensive precision table
    print("\n" + "="*115)
    print(f"{'Model Name':<15} | {'Top-1 %':<9} | {'Top-2 %':<9} (+Δ)    | {'Top-3 %':<9} (+Δ)    | {'Top-4 %':<9} (+Δ)    | {'Top-5 %':<9} (+Δ)")
    print("="*115)
    
    # Trackers for global macro-averages
    history = {k: [] for k in ["top1", "top2", "top3", "top4", "top5"]}
    
    for model in DESIRED_MODEL_ORDER:
        stats = model_stats[model]
        total = stats["total"]
        if total == 0:
            print(f"{model:<15} | No data discovered.")
            continue
            
        p1 = (stats["top1"] / total) * 100
        p2 = (stats["top2"] / total) * 100
        p3 = (stats["top3"] / total) * 100
        p4 = (stats["top4"] / total) * 100
        p5 = (stats["top5"] / total) * 100
        
        history["top1"].append(p1)
        history["top2"].append(p2)
        history["top3"].append(p3)
        history["top4"].append(p4)
        history["top5"].append(p5)
        
        print(f"{model:<15} | {p1:.2f}%   | {p2:.2f}% (+{p2-p1:.2f}%) | {p3:.2f}% (+{p3-p1:.2f}%) | {p4:.2f}% (+{p4-p1:.2f}%) | {p5:.2f}% (+{p5-p1:.2f}%)")
        
    print("="*115)
    if history["top1"]:
        a1 = sum(history["top1"]) / len(history["top1"])
        a2 = sum(history["top2"]) / len(history["top2"])
        a3 = sum(history["top3"]) / len(history["top3"])
        a4 = sum(history["top4"]) / len(history["top4"])
        a5 = sum(history["top5"]) / len(history["top5"])
        print(f"{'AVERAGE':<15} | {a1:.2f}%   | {a2:.2f}% (+{a2-a1:.2f}%) | {a3:.2f}% (+{a3-a1:.2f}%) | {a4:.2f}% (+{a4-a1:.2f}%) | {a5:.2f}% (+{a5-a1:.2f}%)")
    else:
        print("No metrics processed.")
    print("="*115 + "\n")

if __name__ == "__main__":
    calculate_tk_rank_impact()