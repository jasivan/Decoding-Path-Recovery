import os
import json
import glob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean scientific canvas defaults
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 11,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.2
})

OUTPUT_DIR = "inference_eos_uni"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_rank_distributions(data_dir="inference_eos_uni"):
    """Scans JSON logs to calculate the conditional vocabulary rank distribution
       separately for First, Middle, and End token failure types.
    """
    rank_counters = {}
    file_pattern = os.path.join(data_dir, "inference_data_*.json")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace("inference_data_", "").split("_")[0]
        if not model_name or model_name.endswith(".json"):
            model_name = "Qwen3-14B"
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for entry in data:
                metrics = entry.get("metrics", {})
                tokens_block = entry.get("tokens", {})
                outputs_block = entry.get("outputs", {})
                term_analysis = entry.get("termination_analysis", {})
                
                for config in ["base", "ml", "tk"]:
                    config_upper = config.upper()
                    
                    if bool(metrics.get(f"{config}_str_match", False)):
                        continue
                        
                    gold_tokens = tokens_block.get("gold_tokens", [])
                    gen_tokens = tokens_block.get(f"{config}_tokens_generated", [])
                    ranks = tokens_block.get(f"{config}_chosen_vocabulary_ranks", tokens_block.get("tk_chosen_vocabulary_ranks", []))
                    
                    if not gen_tokens or not gold_tokens or not ranks:
                        continue
                        
                    location_type = None
                    fail_rank = None
                    
                    # --- Chronological Step 1: First Token Check ---
                    if gen_tokens[0] != gold_tokens[0] and gen_tokens[0] != -1 and str(gen_tokens[0]) != "-1":
                        if len(ranks) > 0 and ranks[0] != -1 and str(ranks[0]) != "-1":
                            location_type = "First Token"
                            fail_rank = ranks[0]
                            
                    # --- Chronological Step 2: Middle Token Check ---
                    if location_type is None:
                        mismatch_idx = None
                        for i, (g_tok, gen_tok) in enumerate(zip(gold_tokens, gen_tokens)):
                            if gen_tok == -1 or str(gen_tok) == "-1" or g_tok == -1 or str(g_tok) == "-1":
                                continue
                            if g_tok != gen_tok:
                                mismatch_idx = i
                                break
                                
                        if mismatch_idx is not None and mismatch_idx < (len(gold_tokens) - 1):
                            if mismatch_idx < len(ranks) and ranks[mismatch_idx] != -1 and str(ranks[mismatch_idx]) != "-1":
                                location_type = "Middle Token"
                                fail_rank = ranks[mismatch_idx]
                                
                    # --- Chronological Step 3: End Token Check ---
                    if location_type is None:
                        clean_num = outputs_block.get(f"{config}_clean_numeric", "")
                        raw_text = outputs_block.get(f"{config}_raw_text", "")
                        terminated_cleanly = term_analysis.get(f"{config}_terminated_cleanly", True)
                        has_artifact = term_analysis.get(f"{config}_has_prefix_artifact", False)
                        
                        is_end_fail = (not terminated_cleanly) or has_artifact
                        if config == "base" and clean_num and raw_text:
                            if raw_text.strip().startswith(clean_num) and len(raw_text.strip()) > len(clean_num):
                                is_end_fail = True
                                
                        if is_end_fail:
                            final_idx = min(len(gen_tokens) - 1, len(ranks) - 1)
                            if final_idx >= 0 and ranks[final_idx] != -1 and str(ranks[final_idx]) != "-1":
                                location_type = "End Token"
                                fail_rank = ranks[final_idx]
                                
                    if location_type and fail_rank is not None:
                        key = (model_name, config_upper, location_type)
                        if key not in rank_counters:
                            rank_counters[key] = Counter()
                        rank_label = f"Rank {int(fail_rank)}" if int(fail_rank) <= 5 else "Rank >5"
                        rank_counters[key][rank_label] += 1
                        
    return rank_counters

def plot_composite_stacked_ranks(rank_counters):
    """Generates a composite multi-panel grid of stacked bar charts.
       Rows: Failure Location Phases (First, Middle, End)
       Columns: Evaluated Models
       X-Axis: Decoding Configurations (BASE, ML, TK)
    """
    if not rank_counters:
        print("❌ Rank counter logs are empty. Cannot plot profiles.")
        return
        
    records = []
    for (model, config, loc_type), counter in rank_counters.items():
        total_errors = sum(counter.values())
        for rank_lbl, count in counter.items():
            records.append({
                "Model": model,
                "Configuration": config,
                "Location": loc_type,
                "Substituted_Rank": rank_lbl,
                "Percentage": 100.0 * count / total_errors
            })
            
    df = pd.DataFrame(records)
    
    unique_models = sorted(df["Model"].unique())
    location_phases = ["First Token", "Middle Token", "End Token"]
    config_order = ["BASE", "ML", "TK"]
    rank_order = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Rank >5"]
    
    # Establish distinct scientific color palette for vocabulary levels
    palette = sns.color_palette("rocket_r", len(rank_order))
    color_map = {lbl: palette[i] for i, lbl in enumerate(rank_order)}
    
    # Grid setup: Failure Location Phases (Rows) x Models (Columns)
    fig, axes = plt.subplots(len(location_phases), len(unique_models), 
                             figsize=(6.2 * len(unique_models), 4.2 * len(location_phases)), 
                             sharey=True, squeeze=False)
    
    for row_idx, loc in enumerate(location_phases):
        for col_idx, model in enumerate(unique_models):
            ax = axes[row_idx, col_idx]
            
            # Filter rows belonging to this grid tile
            cell_data = df[(df["Model"] == model) & (df["Location"] == loc)]
            
            if cell_data.empty:
                ax.text(0.5, 0.5, "No Deviations Found", ha='center', va='center', color='gray')
                continue
                
            # Pivot data so configurations form rows and ranks form columns
            pivot_df = cell_data.pivot(index="Configuration", columns="Substituted_Rank", values="Percentage")
            pivot_df = pivot_df.reindex(index=config_order, columns=rank_order).fillna(0.0)
            
            pivot_df.plot(
                kind="bar", stacked=True, ax=ax,
                color=[color_map[col] for col in pivot_df.columns],
                edgecolor="#333333", linewidth=1.2, width=0.55, legend=False
            )
            
            ax.set_xlabel("")
            ax.set_xticklabels(config_order, rotation=0, fontweight='bold', fontsize=12)
            ax.set_ylim(0, 105)
            
            # Subplot titles and outer row labels
            if row_idx == 0:
                ax.set_title(f"{model} Profile", fontweight='bold', fontsize=15, pad=12)
            if col_idx == len(unique_models) - 1:
                ax.text(1.05, 0.5, f"{loc} Failures", transform=ax.transAxes, rotation=-90,
                        fontweight='bold', fontsize=14, ha='left', va='center')
                        
            # Place percentage values inside the bar segments
            for rect in ax.patches:
                h = rect.get_height()
                if h > 5.5:  # Only annotate segments tall enough to hold text cleanly
                    ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + h/2., f"{h:.0f}%",
                            ha="center", va="center", color="white", fontweight="bold", fontsize=10)
                            
        axes[row_idx, 0].set_ylabel("Selected Rank Share (%)", fontweight='bold', fontsize=12)
        
    # Construct unified right-aligned legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[r], ec="#333333") for r in rank_order]
    fig.legend(handles, rank_order, loc="center left", bbox_to_anchor=(0.93, 0.5),
               title="Alternative Rank Picked\n(When Rank 0 Was Incorrect)", title_fontsize=12, fontsize=11,
               frameon=True, edgecolor="#cccccc")
    
    plt.subplots_adjust(top=0.91, bottom=0.08, wspace=0.22, hspace=0.32, right=0.87)
    fig.suptitle("Alternative Vocabulary Rank Distributions During Generation Failures\n"
                 "Conditional Probability Profile Over Secondary Token Selections (Summing to 100% per Bar)", 
                 fontsize=17, fontweight='bold', y=0.97)
    
    out_path = os.path.join(OUTPUT_DIR, "global_composite_rank_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Unified composite stacked rank matrix saved to: {out_path}")

if __name__ == "__main__":
    print("Extracting token rank sequences...")
    counters = analyze_rank_distributions("inference_eos_uni")
    plot_composite_stacked_ranks(counters)

# import os
# import glob
# import json
# import re

# # --- CONFIGURATION ---
# DATA_DIR = "inference_eos_uni"

# def clean_model_name(file_path):
#     """Extracts a clean model identification string from the file path."""
#     base = os.path.basename(file_path).replace("inference_data_", "")
#     model_raw = re.sub(r'_(hard|fixed)\.json$', '', base)
#     model_raw = model_raw.split("_")[0]
#     return model_raw

# def analyze_rescue_intersection_steps():
#     file_pattern = os.path.join(DATA_DIR, "inference_data_*.json")
#     files = glob.glob(file_pattern)
    
#     if not files:
#         print(f"No matching files found in directory: '{DATA_DIR}'")
#         return

#     # Master summary pools for both dataset groupings
#     global_pools = {
#         "hard": {"tk": [], "ml": []},
#         "fixed": {"tk": [], "ml": []}
#     }

#     print(f"Found {len(files)} files to analyze.\n")
#     print("Filtering cases where: (tk_str_match == True) AND (ml_str_match == True)\n")
#     print(f"{'Model Name & File Type':<35} | {'Avg TK Gen Step':<16} (Count) | {'Avg ML Gen Step':<16} (Count)")
#     print("-" * 95)

#     for file_path in sorted(files):
#         filename = os.path.basename(file_path)
#         model_name = clean_model_name(file_path)
        
#         if filename.endswith("_hard.json"):
#             suffix_type = "hard"
#             display_name = f"{model_name} (_hard)"
#         elif filename.endswith("_fixed.json"):
#             suffix_type = "fixed"
#             display_name = f"{model_name} (_fixed)"
#         else:
#             continue
        
#         tk_steps_pool = []
#         ml_steps_pool = []
        
#         with open(file_path, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 if isinstance(data, dict): 
#                     data = [data]
#             except (json.JSONDecodeError, UnicodeDecodeError):
#                 continue
            
#             for entry in data:
#                 metrics = entry.get("metrics", {})
                
#                 # --- NEW FILTER CRITERIA ---
#                 # Only analyze steps if BOTH tracks managed to successfully rescue the string match
#                 if metrics.get("tk_str_match") == True and metrics.get("ml_str_match") == True:
                    
#                     # --- TK Step Location Extract ---
#                     tk_ranks = entry.get("tokens", {}).get("tk_chosen_vocabulary_ranks", [])
#                     for gen_step_idx, rank in enumerate(tk_ranks):
#                         if rank is not None and rank > 0:
#                             tk_steps_pool.append(gen_step_idx)
#                             global_pools[suffix_type]["tk"].append(gen_step_idx)
                    
#                     # --- ML Step Location Extract ---
#                     ml_steps = entry.get("layer_traces", {}).get("ml_valid_layers_per_step", [])
#                     for gen_step_idx, layer_list in enumerate(ml_steps):
#                         if isinstance(layer_list, list):
#                             clean_layers = [int(l) for l in layer_list if l not in [None, -1, "-1"]]
#                             if clean_layers and 0 not in clean_layers:
#                                 ml_steps_pool.append(gen_step_idx)
#                                 global_pools[suffix_type]["ml"].append(gen_step_idx)

#         tk_avg_str = f"{(sum(tk_steps_pool) / len(tk_steps_pool)):.2f}" if tk_steps_pool else "N/A"
#         ml_avg_str = f"{(sum(ml_steps_pool) / len(ml_steps_pool)):.2f}" if ml_steps_pool else "N/A"
            
#         print(f"{display_name:<35} | {tk_avg_str:<16} ({len(tk_steps_pool):<5}) | {ml_avg_str:<16} ({len(ml_steps_pool):<5})")

#     # --- GRAND SUMMARY SUFFIX REPORT ---
#     print("\n" + "="*95)
#     print("📊 INTERSECTION RECOVERY SUMMARY BY DATASET SUFFIX (Both Interventions True)")
#     print("="*95)
    
#     for suffix in ["hard", "fixed"]:
#         tk_pool = global_pools[suffix]["tk"]
#         ml_pool = global_pools[suffix]["ml"]
        
#         lbl = "SYMBOLIC EQUATIONS (_hard.json)" if suffix == "hard" else "WORDED QUESTIONS (_fixed.json)"
        
#         tk_final_avg = f"{(sum(tk_pool) / len(tk_pool)):.2f}" if tk_pool else "N/A"
#         ml_final_avg = f"{(sum(ml_pool) / len(ml_pool)):.2f}" if ml_pool else "N/A"
        
#         print(f"📌 {lbl}")
#         print(f"   -> Intersection Avg TK Gen Step : {tk_final_avg:<5} (Total Active Steps: {len(tk_pool)})")
#         print(f"   -> Intersection Avg ML Gen Step : {ml_final_avg:<5} (Total Active Steps: {len(ml_pool)})\n")
#     print("="*95)

# if __name__ == "__main__":
#     analyze_rescue_intersection_steps()