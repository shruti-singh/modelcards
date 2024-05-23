import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from glob import glob

import time
import evaluate
import numpy as np
import pandas as pd
from autoacu import A3CU
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


bertscore = None
bleu = evaluate.load("bleu")
bleurt = None
rscorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calc_rlscore(gen_ans, ans):
    ans = str(ans).strip()
    gen_ans = str(gen_ans).strip()
    scores = rscorer.score(ans, gen_ans)
    return scores['rouge1'].precision*100, scores['rouge2'].precision*100, scores['rougeL'].precision*100

def calc_bleu(predictions, references):
    try:
        results = bleu.compute(predictions=predictions, references=references)
        return results['bleu']*100
    except:
        return 0.0

def calc_bertscore(predictions, references):
    try:
        results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli") #lang="en-sci")
        return results['precision'][0]*100, results['recall'][0]*100, results['f1'][0]*100
    except Exception as ex:
        print("Error during bertscore computation: ", ex)

def calc_bleurtscore(predictions, references):
    results = bleurt.compute(predictions=predictions, references=references)
    return max(results["scores"])*100
