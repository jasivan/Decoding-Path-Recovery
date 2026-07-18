import os
import glob
import json
import re
import pandas as pd
from collections import Counter, defaultdict

# --- GLOBAL DIR SETTINGS ---
DATA_DIR = "inference_eos_uni"
OUTPUT_CSV = os.path.join(DATA_DIR, "model_diagnostic_report.csv")

def clean_model_name(filename):
    base = os.path.basename(filename).replace("inference_data_", "")
    model_raw = re.sub(r'_(hard|fixed)\.json$', '', base)
    model_raw = model_raw.split("_")[0]
    return model_raw.replace("-Instruct", "").replace("-instruct", "")

def parse_baseline_failure(entry):
    gold = str(entry.get("gold_answer", "")).strip()
    base = str(entry.get("outputs", {}).get("base_clean_numeric", "")).strip()
    
    if not gold or not base or gold == base:
        return "None"
    if gold[0] != base[0]:
        return "First Digit Impacted"
        
    if "." in gold and "." in base:
        g_before, g_after = gold.split(".", 1)
        b_before, b_after = base.split(".", 1)
        
        before_wrong = (g_before != b_before)
        after_wrong = (g_after != b_after)
        
        if before_wrong and after_wrong:
            return "Choked Before & After Decimal"
        elif before_wrong:
            return "Choked Before Decimal"
        elif after_wrong:
            return "Choked After Decimal"
            
    return "Precision / Alignment Slip"

def initialize_empty_metrics():
    steps_list = ["First Digit", "Middle Digit", "Final Step", "Trailing Noise Step"]
    step_factory = lambda: {
        f"{step}_{suffix}": 0 
        for step in steps_list 
        for suffix in ["total", "hit"]
    }
    
    context_factory = lambda: {
        "total_base_fails": 0, "tk_wins": 0, "ml_wins": 0,
        "tk_steps": step_factory(), "ml_steps": step_factory()
    }
    
    return {
        "tk": step_factory(),
        "ml": step_factory(),
        "total_rescues_tk": 0,
        "total_rescues_ml": 0,
        "subgroups": defaultdict(context_factory),
        "operations": defaultdict(context_factory),
        "digits": defaultdict(context_factory),
        "baseline_errors": Counter()
    }

def clean_token_string(t):
    if t is None: return ""
    t = str(t)
    return t.replace("Ġ", "").replace(" ", "").replace("Ċ", "").replace("\n", "")

def map_generation_step_to_char_zone(step_idx, tokens_list, gold_str, base_numeric, intervention_len, base_len, matched):
    """
    Maps step indices to true character boundaries, recognizing 'Final Step'
    exclusively when the baseline starts with the gold answer but continues adding digits.
    """
    if step_idx == 0:
        return "First Digit"
        
    prior_str = "".join([clean_token_string(t) for t in tokens_list[:step_idx]])
    prior_len = len(prior_str)
    gold_len = len(gold_str)
    
    # Check if the baseline was a numeric continuation (starts with gold, but longer)
    base_has_extra_digits = base_numeric.startswith(gold_str) and len(base_numeric) > gold_len
    
    if prior_len >= gold_len:
        # Condition: We hit the exact boundary and the baseline failed via extra numbers
        if prior_len == gold_len and base_has_extra_digits and matched and (intervention_len < base_len):
            return "Final Step"
        return "Trailing Noise Step"
        
    if prior_len == 0:
        return "First Digit"
    elif prior_len == gold_len - 1 or (step_idx == len(tokens_list) - 1 and not base_has_extra_digits):
        return "Final Step"
    else:
        return "Middle Digit"

def build_dataset_tables(matrix, all_models):
    error_columns = [
        "First Digit Impacted", "Choked Before Decimal", 
        "Choked After Decimal", "Choked Before & After Decimal", 
        "Precision / Alignment Slip"
    ]
    t1_records, t2_records, t3_records = [], [], []
    
    all_subgroups = sorted(list({sub for m in matrix.values() for sub in m["subgroups"].keys()}))
    all_operations = sorted(list({op for m in matrix.values() for op in m["operations"].keys()}))
    all_digits = sorted(list({dig for m in matrix.values() for dig in m["digits"].keys()}))
    
    steps = ["First Digit", "Middle Digit", "Final Step", "Trailing Noise Step"]

    for model in all_models:
        if model not in matrix: continue
        data = matrix[model]
        
        total_fails = sum(data["baseline_errors"].values())
        t1_row = {
            "Model": model, "Total Base Fails": total_fails,
            "TK Total Rescues": data.get("total_rescues_tk", 0),
            "ML Total Rescues": data.get("total_rescues_ml", 0)
        }
        for err_col in error_columns:
            t1_row[f"Error: {err_col}"] = data["baseline_errors"].get(err_col, 0)
        t1_records.append(t1_row)
        
        t2_row = {"Model": model}
        for track in ["tk", "ml"]:
            lbl = "Non-Zero" if track == "tk" else "Non-Final"
            for step in steps:
                tot = data[track][f"{step}_total"]
                hit = data[track][f"{step}_hit"]
                t2_row[f"Global {track.upper()} {step} ({lbl})"] = f"{(hit/tot*100):.1f}%" if tot > 0 else "0.0%"
        t2_records.append(t2_row)
        
        t3_row = {"Model": model}
        def write_context_metrics(prefix, key_list, structural_dict):
            for item in key_list:
                ctx = structural_dict[item]
                t3_row[f"{prefix}: {item} (TK/ML/Fails)"] = f"{ctx['tk_wins']} / {ctx['ml_wins']} / {ctx['total_base_fails']}"
                for track in ["tk", "ml"]:
                    for step in steps:
                        tot = ctx[f"{track}_steps"][f"{step}_total"]
                        hit = ctx[f"{track}_steps"][f"{step}_hit"]
                        pct = f"{(hit/tot*100):.1f}%" if tot > 0 else "0.0%"
                        t3_row[f"{prefix}: {item} -> {track.upper()} {step}"] = pct

        write_context_metrics("Subgroup", all_subgroups, data["subgroups"])
        write_context_metrics("Operation", all_operations, data["operations"])
        write_context_metrics("Digits Scale", all_digits, data["digits"])
        t3_records.append(t3_row)
        
    return pd.DataFrame(t1_records), pd.DataFrame(t2_records), pd.DataFrame(t3_records)

def generate_csv_report():
    file_pattern = os.path.join(DATA_DIR, "inference_data_*.json")
    files = glob.glob(file_pattern)
    hard_matrix, fixed_matrix = {}, {}   
    
    for file_path in files:
        filename = os.path.basename(file_path)
        model_key = clean_model_name(file_path)
        
        if filename.endswith("_hard.json"): target_matrix = hard_matrix
        elif filename.endswith("_fixed.json"): target_matrix = fixed_matrix
        else: continue 
            
        if model_key not in target_matrix:
            target_matrix[model_key] = initialize_empty_metrics()
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
                
            for entry in data:
                metrics = entry.get("metrics", {})
                if not metrics.get("base_str_match", True):
                    model_data = target_matrix[model_key]
                    
                    err_signature = parse_baseline_failure(entry)
                    model_data["baseline_errors"][err_signature] += 1
                    
                    subgroup = entry.get("subgroup", "unknown")
                    op = entry.get("operation", "unknown")
                    digits = entry.get("gold_digit_count", 0)
                    
                    ctx_pointers = [
                        model_data["subgroups"][subgroup],
                        model_data["operations"][op],
                        model_data["digits"][digits]
                    ]
                    for ctx in ctx_pointers: ctx["total_base_fails"] += 1
                    
                    gold_str = str(entry.get("gold_answer", "")).strip()
                    base_numeric = str(entry.get("outputs", {}).get("base_clean_numeric", "")).strip()
                    base_tokens = entry.get("tokens", {}).get("base_tokens_generated", [])
                    base_len = len(base_tokens)
                    
                    base_has_extra_digits = base_numeric.startswith(gold_str) and len(base_numeric) > len(gold_str)

                    # --- TK TRACKING ROUTINE ---
                    tk_matched = (metrics.get("tk_str_match") == True)
                    tk_ranks = entry.get("tokens", {}).get("tk_chosen_vocabulary_ranks", [])
                    tk_tokens = entry.get("tokens", {}).get("tk_tokens_generated", [])
                    tk_len = len(tk_ranks)
                    
                    if tk_matched:
                        model_data["total_rescues_tk"] += 1
                        for ctx in ctx_pointers: ctx["tk_wins"] += 1
                            
                    for step_idx, rank in enumerate(tk_ranks):
                        step_lbl = map_generation_step_to_char_zone(
                            step_idx, tk_tokens, gold_str, base_numeric, tk_len, base_len, tk_matched
                        )
                        is_hit = (rank is not None and rank > 0)
                        
                        if step_lbl == "Final Step" and tk_matched and base_has_extra_digits and (tk_len < base_len):
                            is_hit = True

                        model_data["tk"][f"{step_lbl}_total"] += 1
                        if is_hit: model_data["tk"][f"{step_lbl}_hit"] += 1
                        for ctx in ctx_pointers:
                            ctx["tk_steps"][f"{step_lbl}_total"] += 1
                            if is_hit: ctx["tk_steps"][f"{step_lbl}_hit"] += 1
                                
                    # --- ML TRACKING ROUTINE ---
                    ml_matched = (metrics.get("ml_str_match") == True)
                    ml_layers_per_step = entry.get("layer_traces", {}).get("ml_valid_layers_per_step", [])
                    ml_tokens = entry.get("tokens", {}).get("ml_tokens_generated", [])
                    ml_len = len(ml_layers_per_step)
                    
                    if ml_matched:
                        model_data["total_rescues_ml"] += 1
                        for ctx in ctx_pointers: ctx["ml_wins"] += 1
                            
                    for step_idx, layers in enumerate(ml_layers_per_step):
                        if not isinstance(layers, list): continue
                        
                        step_lbl = map_generation_step_to_char_zone(
                            step_idx, ml_tokens, gold_str, base_numeric, ml_len, base_len, ml_matched
                        )
                        clean_layers = [int(l) for l in layers if l not in [None, -1, "-1"]]
                        is_hit = (len(clean_layers) > 0 and 0 not in clean_layers and "0" not in clean_layers)
                        
                        if step_lbl == "Final Step" and ml_matched and base_has_extra_digits and (ml_len < base_len):
                            is_hit = True

                        model_data["ml"][f"{step_lbl}_total"] += 1
                        if is_hit: model_data["ml"][f"{step_lbl}_hit"] += 1
                        for ctx in ctx_pointers:
                            ctx["ml_steps"][f"{step_lbl}_total"] += 1
                            if is_hit: ctx["ml_steps"][f"{step_lbl}_hit"] += 1

    all_models = sorted(list(set(list(hard_matrix.keys()) + list(fixed_matrix.keys()))))
    hard_t1, hard_t2, hard_t3 = build_dataset_tables(hard_matrix, all_models)
    fixed_t1, fixed_t2, fixed_t3 = build_dataset_tables(fixed_matrix, all_models)

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        def write_section(title, df):
            f.write(f"\n# === {title.upper()} ===\n")
            df.to_csv(f, index=False)
            f.write("\n")

        f.write("# MODEL BEHAVIOR DIAGNOSTIC MASTER REPORT\n")
        f.write("\n" + "="*60 + "\n# PART 1: SYMBOLIC EQUATIONS (_hard)\n" + "="*60 + "\n")
        write_section("1. Baseline Failures and Total Rescues", hard_t1)
        write_section("2. Global Generational Step Dynamics", hard_t2)
        write_section("3. Detailed Context Cohorts", hard_t3)
        
        f.write("\n" + "="*60 + "\n# PART 2: WORDED QUESTIONS (_fixed)\n" + "="*60 + "\n")
        write_section("1. Baseline Failures and Total Rescues", fixed_t1)
        write_section("2. Global Generational Step Dynamics", fixed_t2)
        write_section("3. Detailed Context Cohorts", fixed_t3)

    print(f"🎉 Complete! Numeric-continuation scoped report generated at: {OUTPUT_CSV}\n")

if __name__ == "__main__":
    generate_csv_report()