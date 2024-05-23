import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from glob import glob

import evaluate
import numpy as np
import pandas as pd
# from autoacu import A3CU
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


bertscore = None
bleu = evaluate.load("bleu")
bleurt = None
rscorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

skip_metrics = []

def calc_rlscore(gen_ans, ans):
    gen_ans_list = load_gen_strings_byeval(gen_ans)
    ans = str(ans).strip()
    rscores = {"r1": [], "r2": [], "rl": []}
    for ga in gen_ans_list:
        gen_ans = str(ga).strip()
        scores = rscorer.score(ans, gen_ans)
        rscores['r1'].append(scores['rouge1'].precision)
        rscores['r2'].append(scores['rouge2'].precision)
        rscores['rl'].append(scores['rougeL'].precision)
    score_list = []
    for r in ["r1", "r2", "rl"]:
        score_list.append(max(rscores[r], default=0))
    return score_list[0]*100, score_list[1]*100, score_list[2]*100

def calc_bleu(predictions, references):
    predictions_list = load_gen_strings_byeval(predictions)
    bleu_score_list = []
    for pred in predictions_list:
        try:
            results = bleu.compute(predictions=[pred], references=references)
            bleu_score_list.append(results['bleu'])
        except:
            bleu_score_list.append(0)
    return max(bleu_score_list, default=0)*100

def calc_bertscore(predictions, references):
    predictions_list = load_gen_strings_byeval(predictions)
    ref_list = [references]*len(predictions_list)
    # print(len(predictions_list), len(ref_list))
    # print("PREDICTIONS: ",predictions)
    # print("REFERENCE LIST: ",ref_list)
    results = bertscore.compute(predictions=predictions_list, references=ref_list, model_type="microsoft/deberta-xlarge-mnli", batch_size=64) 
    return max(results['precision'], default=0)*100, max(results['recall'], default=0)*100, max(results['f1'], default=0)*100

def calc_bleurtscore(predictions, references):
    predictions_list = load_gen_strings_byeval(predictions)
    ref_list = [references]*len(predictions_list)
    # print(len(predictions_list), len(ref_list))
    results = bleurt.compute(predictions=predictions_list, references=ref_list)
    return max(results["scores"], default=0)*100


all_metrics = ["r1", "r2", "rl", "bleuscore", "bleurt", "bs_precision", "bs_recall", "bs_f1"]


target_file_prefix = "./results/alt_manuscript_version" 
for config, fpath in zip(["ft/alt_manuscript_version"], [f"./llm_resp/ft/vicuna_7b.xlsx"]):

    cons_df = pd.read_excel(fpath)
    
    all_model_names = []
    for i in list(cons_df.columns):
        if i.find("_ans") > -1:
            all_model_names.append(i)

    for resp_file in all_resp_files[1:]:
        resp_df = pd.read_excel(resp_file)
        resp_df = resp_df.sort_values('id')
        for i in list(resp_df.columns):
            if i.find("_ans") > -1:
                model_ans_col = i
                all_model_names.append(i)
                break
        cons_df[model_ans_col] = resp_df[model_ans_col]

  
    bleurt = load("bleurt", "BLEURT-20")
    print("Calculating bleurt score...")
    for mname in all_model_names:
        print(mname)
        if f"{mname}_bleurt" in skip_metrics:
            cons_df[f"{mname}_bleurt"] = org_df[f"{mname}_bleurt"]
            continue
        dd = cons_df.apply(lambda x: calc_bleurtscore(x[mname], x.ans), axis=1)
        cons_df[f"{mname}_bleurt"] = list(np.array([*dd.values]))
        cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")
    bleurt = None
    cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")

    print("Calculating rouge...")
    for mname in all_model_names:
        print(mname)
        if f"{mname}_r1" in skip_metrics and f"{mname}_r2" in skip_metrics and f"{mname}_rl" in skip_metrics:
            cons_df[f"{mname}_r1"] = org_df[f"{mname}_r1"]
            cons_df[f"{mname}_r2"] = org_df[f"{mname}_r2"]
            cons_df[f"{mname}_rl"] = org_df[f"{mname}_rl"]
            continue
        dd = cons_df.apply(lambda x: calc_rlscore(x[mname], x.ans), axis=1)
        cons_df[f"{mname}_r1"] = list(np.array([*dd.values])[:,0])
        cons_df[f"{mname}_r2"] = list(np.array([*dd.values])[:,1])
        cons_df[f"{mname}_rl"] = list(np.array([*dd.values])[:,2])
        cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")
    cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")

    print("Calculating bleu...")
    for mname in all_model_names:
        print(mname)
        if f"{mname}_bleuscore" in skip_metrics:
            cons_df[f"{mname}_bleuscore"] = org_df[f"{mname}_bleuscore"]
            continue
        cons_df[f"{mname}_bleuscore"] = cons_df.apply(lambda x: calc_bleu(x[mname], [[x.ans]]), axis=1)
        cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")
    cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")

    print("Calculating bertscore...")
    bertscore = load("bertscore", model_type="microsoft/deberta-xlarge-mnli")
    for mname in all_model_names:
        print(mname)
        if f"{mname}_bs_precision" in skip_metrics and f"{mname}_bs_recall" in skip_metrics and f"{mname}_bs_f1" in skip_metrics:
            cons_df[f"{mname}_bs_precision"] = org_df[f"{mname}_bs_precision"]
            cons_df[f"{mname}_bs_recall"] = org_df[f"{mname}_bs_recall"]
            cons_df[f"{mname}_bs_f1"] = org_df[f"{mname}_bs_f1"]
            continue
        cons_df[mname] = cons_df[mname].astype(str)
        try:
            dd = cons_df.apply(lambda x: calc_bertscore(x[mname], x.ans), axis=1)
            cons_df[f"{mname}_bs_precision"] = list(np.array([*dd.values])[:,0])
            cons_df[f"{mname}_bs_recall"] = list(np.array([*dd.values])[:,1])
            cons_df[f"{mname}_bs_f1"] = list(np.array([*dd.values])[:,2])
        except:
            pass
        cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")
    bertscore = None
    cons_df.to_excel(f"{target_file_prefix}/{config}_eval_scores_{run_id}_final.xlsx")

    time.sleep(3)
    
    print("\n\n")
    print(fpath)
    for mname in sorted(all_model_names):
        model_metrics = []
        for metric in [ "r1", "r2", "rl", "bleurt", "bs_f1"]:
            if f"{mname}_{metric}" in cons_df.columns:
                model_metrics.append(round(cons_df[f"{mname}_{metric}"].mean(), 3))
        print(", ".join([mname] + [str(x) for x in model_metrics]))

