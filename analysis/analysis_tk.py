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