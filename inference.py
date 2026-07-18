import argparse
import json
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Pure Greedy Multi-Layer Horizon Matrix - Fully Instrumented & Optimized")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="FERMAT_all.json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--type", type=str, default="question")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Optional explicit output path. If None, it will be generated dynamically.")
    args = parser.parse_args()
    
    if args.output_file is None:
        model_clean = args.model.split("/")[-1]
        dataset_clean = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_file = f"inference_eos/inference_data_{model_clean}_{dataset_clean}_{args.type}.json"
        
    return args

def determine_subgroup_by_index(current_idx):
    bin_size = 531
    bin_number = current_idx // bin_size
    subgroups = ["2 digit", "3 digit", "4 digit", "4+ digit", "1dp", "2dp", "4+ digit hard", "1dp random", "2dp random"]
    return subgroups[bin_number] if bin_number < len(subgroups) else "2dp random"

def determine_operation_type(equation_str):
    if "+" in equation_str: return "addition"
    elif "/" in equation_str or "÷" in equation_str: return "division"
    elif "*" in equation_str or "x" in equation_str or "×" in equation_str: return "multiplication"
    elif "-" in equation_str: return "subtraction"
    return "unknown"

def extract_clean_digits(text):
    return "".join([c for c in text if c.isdigit() or c in [".", "-"]])

def format_prompt_by_type(raw_equation_str):
    cleaned = raw_equation_str.strip()
    if re.search(r'[a-zA-Z]{2,}', cleaned): return f"Give the final answer only. {cleaned} Answer: "
    if cleaned.startswith("(") and cleaned.endswith(")"): cleaned = cleaned[1:-1].strip()
    return f"Give the final answer only. {cleaned} = "

def apply_nucleus_filter(logits, top_k_window=20, top_p=0.8, temperature=0.7):
    scaled_logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(scaled_logits, k=top_k_window)
    filter_mask = torch.ones_like(scaled_logits) * float("-inf")
    filter_mask[top_k_indices] = top_k_values
    probs = F.softmax(filter_mask, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    filter_mask[sorted_indices[sorted_indices_to_remove]] = float("-inf")
    return filter_mask

def calculate_prefix_token_penalty(generated_tokens, gold_tokens):
    match_count = 0
    for g_t, gold_t in zip(generated_tokens, gold_tokens):
        if g_t == gold_t: match_count += 1
        else: break
    return match_count, max(1, len(gold_tokens))

def calculate_digit_accuracy(generated_str, gold_str):
    match_count = 0
    for g_c, gold_c in zip(generated_str, gold_str):
        if g_c == gold_c: match_count += 1
        else: break
    return match_count, max(1, len(gold_str))

def build_numeric_vocabulary_mask(tokenizer, model, device):
    vocab_size = model.config.vocab_size
    mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for token_str, token_id in tokenizer.get_vocab().items():
        if token_id >= vocab_size: continue
        decoded_str = tokenizer.decode([token_id])
        if any(c.isdigit() for c in decoded_str):
            mask[token_id] = 1.0
    return mask

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if device=="cuda" else torch.float32, trust_remote_code=True).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    early_exit_layers = list(range(1, num_layers))
    with open(args.dataset, "r", encoding="utf-8") as f: lines = [line.strip() for line in f if line.strip()]

    print("Pre-indexing vocabulary tokens to flag numeric identities...")
    numeric_mask = build_numeric_vocabulary_mask(tokenizer, model, device)
    mask_len = numeric_mask.shape[0]

    numeric_vocab_dump = {}
    for token_str, token_id in tokenizer.get_vocab().items():
        if token_id < mask_len and numeric_mask[token_id].item() == 1.0:
            numeric_vocab_dump[int(token_id)] = tokenizer.decode([token_id])
            
    vocab_output_file = args.output_file.replace("inference_data_", "numeric_vocab_")
    print(f"Saving compiled numeric vocabulary matrix to: {vocab_output_file}")
    with open(vocab_output_file, "w", encoding="utf-8") as f:
        json.dump(numeric_vocab_dump, f, indent=2, ensure_ascii=False, sort_keys=True)

    subgroups = ["2 digit", "3 digit", "4 digit", "4+ digit", "1dp", "2dp", "4+ digit hard", "1dp random", "2dp random"]
    subgroup_metrics = {
        sg: {
            "total": 0, "base_success": 0, "multi_layer_str_success": 0, "top_k_str_success": 0,
            "base_digit_num": 0, "base_digit_den": 0, "ml_digit_num": 0, "ml_digit_den": 0, "tk_digit_num": 0, "tk_digit_den": 0,
            "base_tok_num": 0, "base_tok_den": 0, "ml_tok_num": 0, "ml_tok_den": 0, "tk_tok_num": 0, "tk_tok_den": 0
        } for sg in subgroups
    }
    
    total_valid_items = 0
    eval_results, valid_item_index = [], 0

    for line in tqdm(lines, desc="Evaluating"):
        try:
            data = json.loads(line)
            raw_prompt, raw_eqn, true_ans_str = data[args.type], data["equation"], str(data["answer"]).strip()
        except: continue

        sg_type = determine_subgroup_by_index(valid_item_index)
        op_type = determine_operation_type(raw_eqn)
        gold_digits_len = len("".join([c for c in true_ans_str if c.isdigit()]))

        valid_item_index += 1
        total_valid_items += 1
        subgroup_metrics[sg_type]["total"] += 1

        formatted_prompt = format_prompt_by_type(raw_prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        gold_token_ids = tokenizer.encode(true_ans_str, add_special_tokens=False)
        base_prompt_ids = inputs["input_ids"]

        true_ans_no_comma = true_ans_str.replace(",", "")

        # -------------------------------------------------------------
        # 1. CLEAN ROUTE A: NATIVE GENERATION
        # -------------------------------------------------------------
        with torch.no_grad():
            base_out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, 
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id
            )
        
        base_text = tokenizer.decode(base_out[0][base_prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        base_tokens_generated = base_out[0][base_prompt_ids.shape[1]:].tolist()
        first_numeric_match = re.search(r'[-+]?\d*[\.,]?\d+', base_text)
        base_clean_ans = first_numeric_match.group(0) if first_numeric_match else ""
        base_clean_ans = base_clean_ans.strip().lstrip("+")
        
        base_clean_no_comma = base_clean_ans.replace(",", "")
        is_base_correct = (base_clean_no_comma == true_ans_no_comma)
        if is_base_correct: subgroup_metrics[sg_type]["base_success"] += 1

        # -------------------------------------------------------------
        # 2. STEP-BY-STEP CUSTOM LOOPS
        # -------------------------------------------------------------
        ml_step_ids, tk_step_ids = base_prompt_ids.clone(), base_prompt_ids.clone()
        ml_tokens_generated, tk_tokens_generated = [], []
        ml_generated_str, tk_generated_str = "", ""
        
        ml_valid_layers_per_step, ml_numeric_prob_mass_per_step, ml_layer_predictions_fine_grain = [], [], []
        tk_chosen_ranks_per_step, tk_predictions_per_step = [], []

        ml_diverged, tk_diverged = False, False
        ml_failure_step, tk_failure_step = -1, -1
        
        ml_terminating_layers = None
        tk_terminating_layers = None
        
        ml_past_key_values, tk_past_key_values = None, None

        for step_idx in range(args.max_new_tokens):
            ml_done = ml_generated_str.startswith(true_ans_no_comma) or len(ml_generated_str) >= len(true_ans_no_comma) + 2
            tk_done = tk_generated_str.startswith(true_ans_no_comma) or len(tk_generated_str) >= len(true_ans_no_comma) + 2
            if ml_done and tk_done: break

            # --- ROUTE B: PURE GREEDY MULTI-LAYER ---
            if not ml_done:
                if ml_diverged:
                    with torch.no_grad(): 
                        ml_outputs = model(input_ids=ml_step_ids[:, -1:], past_key_values=ml_past_key_values, use_cache=True)
                    ml_past_key_values = ml_outputs.past_key_values
                    l0_logits = ml_outputs.logits[0, -1, :].float()
                    ml_chosen_token_id = torch.argmax(l0_logits).item()
                    
                    ml_valid_layers_per_step.append([])
                    l0_probs = F.softmax(l0_logits, dim=-1)
                    ml_numeric_prob_mass_per_step.append({"layer_0": round(torch.sum(l0_probs[:mask_len] * numeric_mask).item(), 5)})
                    ml_layer_predictions_fine_grain.append({}) 
                
                else:
                    with torch.no_grad(): 
                        ml_outputs = model(
                            input_ids=ml_step_ids if step_idx == 0 else ml_step_ids[:, -1:], 
                            past_key_values=ml_past_key_values, use_cache=True, output_hidden_states=True
                        )
                    ml_past_key_values = ml_outputs.past_key_values
                    
                    valid_layers_this_step, prob_mass_this_step, fine_grain_this_step = [], {}, {}
                    l0_logits = ml_outputs.logits[0, -1, :].float()
                    
                    hidden_stack = torch.stack([ml_outputs.hidden_states[l + 1][0, -1, :] for l in early_exit_layers])
                    all_early_logits = model.lm_head(hidden_stack.to(model.lm_head.weight.dtype)).float()
                    
                    layers_to_check = [("layer_0", l0_logits, 0)]
                    for idx, l_idx in enumerate(early_exit_layers):
                        layers_to_check.append((f"layer_{l_idx}", all_early_logits[idx], l_idx))

                    predicted_chosen_id = torch.argmax(layers_to_check[0][1]).item()
                    temp_best_len = -1
                    l0_c = extract_clean_digits(tokenizer.decode([predicted_chosen_id]))
                    if l0_c != "" and true_ans_no_comma.startswith(ml_generated_str + l0_c):
                        temp_best_len = len(l0_c)
                    
                    for name, logits, l_idx in layers_to_check[1:]:
                        pred_id = torch.argmax(logits).item()
                        c_clean = extract_clean_digits(tokenizer.decode([pred_id]))
                        if c_clean != "" and true_ans_no_comma.startswith(ml_generated_str + c_clean):
                            if len(c_clean) >= temp_best_len:
                                temp_best_len = len(c_clean)
                                predicted_chosen_id = pred_id

                    layers_with_valid_stop_this_step = []
                    for name, logits, l_idx in layers_to_check:
                        probs = F.softmax(logits, dim=-1)
                        prob_mass_this_step[name] = round(torch.sum(probs[:mask_len] * numeric_mask).item(), 5)
                        
                        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                        chosen_rank = (sorted_ids == predicted_chosen_id).nonzero(as_tuple=True)[0][0].item()
                        chosen_prob = probs[predicted_chosen_id].item()

                        absolute_top_2 = []
                        for rank_idx in range(2):
                            absolute_top_2.append({
                                "token": tokenizer.decode([sorted_ids[rank_idx].item()]),
                                "prob": round(sorted_probs[rank_idx].item(), 5),
                                "rank": rank_idx
                            })

                        gold_token_info = {"token": "", "prob": 0.0, "rank": -1}
                        top_slice_probs = sorted_probs[:500].tolist()
                        top_slice_ids = sorted_ids[:500].tolist()
                        
                        for rank, (tok_id, tok_prob) in enumerate(zip(top_slice_ids, top_slice_probs)):
                            tok_str = tokenizer.decode([tok_id])
                            tok_clean = extract_clean_digits(tok_str)
                            if tok_clean != "" and true_ans_no_comma.startswith(ml_generated_str + tok_clean):
                                gold_token_info = {"token": tok_str, "prob": round(tok_prob, 5), "rank": rank}
                                break

                        top_layer_id = torch.argmax(logits).item()
                        top_layer_clean = extract_clean_digits(tokenizer.decode([top_layer_id]))
                        
                        is_valid_termination = (ml_generated_str == true_ans_no_comma) and (top_layer_id == tokenizer.eos_token_id or top_layer_clean == "")
                        if is_valid_termination:
                            layers_with_valid_stop_this_step.append(l_idx)

                        if top_layer_clean != "" and true_ans_no_comma.startswith(ml_generated_str + top_layer_clean):
                            valid_layers_this_step.append(l_idx)

                        fine_grain_this_step[name] = {
                            "chosen_token": {"token": tokenizer.decode([predicted_chosen_id]), "prob": round(chosen_prob, 5), "rank": chosen_rank},
                            "absolute_top_2_predictions": absolute_top_2,
                            "gold_token_tracking": gold_token_info
                        }

                    if ml_generated_str == true_ans_no_comma and ml_terminating_layers is None:
                        ml_terminating_layers = layers_with_valid_stop_this_step

                    ml_chosen_token_id = predicted_chosen_id
                    ml_valid_layers_per_step.append(valid_layers_this_step)
                    ml_numeric_prob_mass_per_step.append(prob_mass_this_step)
                    ml_layer_predictions_fine_grain.append(fine_grain_this_step)
                    
                    if not valid_layers_this_step and not (ml_generated_str == true_ans_no_comma):
                        ml_diverged = True
                        ml_failure_step = step_idx

                ml_tokens_generated.append(ml_chosen_token_id)
                ml_generated_str += extract_clean_digits(tokenizer.decode([ml_chosen_token_id]))
                ml_step_ids = torch.cat([ml_step_ids, torch.tensor([[ml_chosen_token_id]], device=device)], dim=-1)

            # --- ROUTE C: TOP-K ---
            if not tk_done:
                with torch.no_grad(): 
                    tk_outputs = model(input_ids=tk_step_ids if step_idx == 0 else tk_step_ids[:, -1:], past_key_values=tk_past_key_values, use_cache=True)
                tk_past_key_values = tk_outputs.past_key_values
                
                tk_logits = tk_outputs.logits[0, -1, :]
                tk_vocab_probs = F.softmax(tk_logits, dim=-1)
                sorted_tk_probs, sorted_tk_ids = torch.sort(tk_vocab_probs, descending=True)
                
                if tk_diverged:
                    tk_chosen_token_id = torch.argmax(tk_logits).item()
                    chosen_rank_in_vocab = (sorted_tk_ids == tk_chosen_token_id).nonzero(as_tuple=True)[0][0].item()
                    tk_chosen_ranks_per_step.append(chosen_rank_in_vocab)
                    tk_predictions_per_step.append([])
                else:
                    tk_filtered = apply_nucleus_filter(tk_logits)
                    top_k_preds = torch.topk(tk_filtered, k=args.top_k).indices.tolist()
                    
                    # Log metrics for ALL Top-K options extracted from the final prediction layer
                    step_top_k_details = []
                    for rank_idx, pred_id in enumerate(top_k_preds):
                        vocab_rank = (sorted_tk_ids == pred_id).nonzero(as_tuple=True)[0][0].item()
                        step_top_k_details.append({
                            "token": tokenizer.decode([pred_id]),
                            "rank": vocab_rank,
                            "prob": round(tk_vocab_probs[pred_id].item(), 5)
                        })
                    tk_predictions_per_step.append(step_top_k_details)

                    tk_chosen_token_id = top_k_preds[0]
                    found_valid_tk = False
                    
                    for rank_idx, pred_id in enumerate(top_k_preds):
                        pred_clean = extract_clean_digits(tokenizer.decode([pred_id]))
                        if pred_clean != "" and true_ans_no_comma.startswith(tk_generated_str + pred_clean):
                            tk_chosen_token_id = pred_id
                            found_valid_tk = True
                            break
                    
                    chosen_rank_in_vocab = (sorted_tk_ids == tk_chosen_token_id).nonzero(as_tuple=True)[0][0].item()
                    tk_chosen_ranks_per_step.append(chosen_rank_in_vocab)

                    if tk_generated_str == true_ans_no_comma and tk_terminating_layers is None:
                        top_tk_clean = extract_clean_digits(tokenizer.decode([top_k_preds[0]]))
                        tk_terminating_layers = [0] if (top_k_preds[0] == tokenizer.eos_token_id or top_tk_clean == "") else []

                    if not found_valid_tk and not (tk_generated_str == true_ans_no_comma):
                        tk_diverged = True
                        tk_failure_step = step_idx

                tk_tokens_generated.append(tk_chosen_token_id)
                tk_generated_str += extract_clean_digits(tokenizer.decode([tk_chosen_token_id]))
                tk_step_ids = torch.cat([tk_step_ids, torch.tensor([[tk_chosen_token_id]], device=device)], dim=-1)

        # -------------------------------------------------------------
        # METRICS RECONCILIATION & TERMINATION VERIFICATION
        # -------------------------------------------------------------
        ml_str_success = (ml_generated_str == true_ans_no_comma) if true_ans_no_comma else False
        top_k_str_success = (tk_generated_str == true_ans_no_comma) if true_ans_no_comma else False
        
        ml_terminated_cleanly = ml_str_success and (ml_tokens_generated[-1] == tokenizer.eos_token_id or extract_clean_digits(tokenizer.decode([ml_tokens_generated[-1]])) == "")
        tk_terminated_cleanly = top_k_str_success and (tk_tokens_generated[-1] == tokenizer.eos_token_id or extract_clean_digits(tokenizer.decode([tk_tokens_generated[-1]])) == "")

        ml_has_prefix_artifact = ml_str_success and len(ml_tokens_generated) > 0 and bool(re.match(r'^\D', tokenizer.decode([ml_tokens_generated[0]])))
        tk_has_prefix_artifact = top_k_str_success and len(tk_tokens_generated) > 0 and bool(re.match(r'^\D', tokenizer.decode([tk_tokens_generated[0]])))

        if ml_str_success: subgroup_metrics[sg_type]["multi_layer_str_success"] += 1
        if top_k_str_success: subgroup_metrics[sg_type]["top_k_str_success"] += 1

        for route, g_str, t_gen in [("base", base_clean_ans, base_tokens_generated), ("ml", ml_generated_str, ml_tokens_generated), ("tk", tk_generated_str, tk_tokens_generated)]:
            dn, dd = calculate_digit_accuracy(g_str, true_ans_no_comma)
            tn, td = calculate_prefix_token_penalty(t_gen, gold_token_ids)
            subgroup_metrics[sg_type][f"{route}_digit_num"] += dn; subgroup_metrics[sg_type][f"{route}_digit_den"] += dd
            subgroup_metrics[sg_type][f"{route}_tok_num"] += tn; subgroup_metrics[sg_type][f"{route}_tok_den"] += td

        eval_results.append({
            "prompt": raw_prompt, 
            "gold_answer": true_ans_str, 
            "gold_digit_count": gold_digits_len,
            "subgroup": sg_type, 
            "operation": op_type,
            "failure_points": {
                "ml_failure_step": ml_failure_step, 
                "tk_failure_step": tk_failure_step
            },
            "termination_analysis": {
                "ml_terminated_cleanly": ml_terminated_cleanly,
                "ml_has_prefix_artifact": ml_has_prefix_artifact,
                "ml_valid_terminating_layers": ml_terminating_layers if ml_terminating_layers is not None else [],
                "tk_terminated_cleanly": tk_terminated_cleanly,
                "tk_has_prefix_artifact": tk_has_prefix_artifact,
                "tk_valid_terminating_layers": tk_terminating_layers if tk_terminating_layers is not None else []
            },
            "metrics": {
                "base_str_match": is_base_correct, 
                "ml_str_match": ml_str_success, 
                "tk_str_match": top_k_str_success
            },
            "outputs": {
                "base_raw_text": base_text, 
                "base_clean_numeric": base_clean_ans, 
                "ml_generated_str": ml_generated_str, 
                "tk_generated_str": tk_generated_str
            },
            "tokens": {
                "gold_tokens": tokenizer.convert_ids_to_tokens(gold_token_ids),
                "base_tokens_generated": tokenizer.convert_ids_to_tokens(base_tokens_generated),
                "ml_tokens_generated": tokenizer.convert_ids_to_tokens(ml_tokens_generated),
                "tk_tokens_generated": tokenizer.convert_ids_to_tokens(tk_tokens_generated),
                "tk_chosen_vocabulary_ranks": tk_chosen_ranks_per_step,
                "tk_step_predictions": tk_predictions_per_step
            },
            "layer_traces": {
                "ml_valid_layers_per_step": ml_valid_layers_per_step,
                "ml_numeric_prob_mass_per_step": ml_numeric_prob_mass_per_step,
                "ml_layer_predictions_fine_grain": ml_layer_predictions_fine_grain
            }
        })

    # =================================================================
    # SUMMARY DISPLAY TABLE
    # =================================================================
    print("\n" + "=" * 135)
    print("               EXPANDED ARITHMETIC PERFORMANCE MATRIX (TOKEN COHESION MATRIX ALIGNED)")
    print("=" * 135)
    print(f"{'Subgroup':16} | {'Base Str':8} | {'Base Dig':8} | {'Base Tok*':9} | {'ML Str':7} | {'ML Dig':7} | {'ML Tok*':8} | {f'Top-{args.top_k} Str':9} | {f'Top-{args.top_k} Dig':9} | {f'Top-{args.top_k} Tok*':10}")
    print("-" * 135)
    
    g = {k: 0 for k in ["tot", "bs", "bdn", "bdd", "btn", "btd", "ms", "mdn", "mdd", "mtn", "mtd", "ts", "tdn", "tdd", "ttn", "ttd"]}
    for sg in subgroups:
        m = subgroup_metrics[sg]; tot = m["total"]
        if tot == 0: continue
        g["tot"] += tot; g["bs"] += m["base_success"]; g["ms"] += m["multi_layer_str_success"]; g["ts"] += m["top_k_str_success"]
        g["bdn"] += m["base_digit_num"]; g["bdd"] += m["base_digit_den"]; g["mdn"] += m["ml_digit_num"]; g["mdd"] += m["ml_digit_den"]; g["tdn"] += m["tk_digit_num"]; g["tdd"] += m["tk_digit_den"]
        g["btn"] += m["base_tok_num"]; g["btd"] += m["base_tok_den"]; g["mtn"] += m["ml_tok_num"]; g["mtd"] += m["ml_tok_den"]; g["ttn"] += m["tk_tok_num"]; g["ttd"] += m["tk_tok_den"]

        print(f"{sg:16} | {(m['base_success']/tot)*100:6.1f}% | {(m['base_digit_num']/m['base_digit_den'])*100:6.1f}% | {(m['base_tok_num']/m['base_tok_den'])*100:7.1f}% | {(m['multi_layer_str_success']/tot)*100:5.1f}% | {(m['ml_digit_num']/m['ml_digit_den'])*100:5.1f}% | {(m['ml_tok_num']/m['ml_tok_den'])*100:6.1f}% | {(m['top_k_str_success']/tot)*100:7.1f}% | {(m['tk_digit_num']/m['tk_digit_den'])*100:7.1f}% | {(m['tk_tok_num']/m['tk_tok_den'])*100:8.1f}%")
    print("-" * 135)
    print(f"{'GLOBAL MEAN':16} | {(g['bs']/g['tot'])*100:6.1f}% | {(g['bdn']/g['bdd'])*100:6.1f}% | {(g['btn']/g['btd'])*100:7.1f}% | {(g['ms']/g['tot'])*100:5.1f}% | {(g['mdn']/g['mdd'])*100:5.1f}% | {(g['mtn']/g['mtd'])*100:6.1f}% | {(g['ts']/g['tot'])*100:7.1f}% | {(g['tdn']/g['tdd'])*100:7.1f}% | {(g['ttn']/g['ttd'])*100:8.1f}%")
    print("=" * 135)
    
    print(f"Saving fully instrumented compiled matrix: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f: json.dump(eval_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__": main()