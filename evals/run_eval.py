import os
from eval_auto import calc_rlscore, calc_bleu, calc_bertscore, calc_bleurtscore
import pandas as pd
import numpy as np
import time
import sys

RESULTS_SAVE_PATH = "/home/xhitij.cm/modelcards/evaluation_results"
if not os.path.exists(RESULTS_SAVE_PATH):
    os.mkdir(RESULTS_SAVE_PATH)

class RunEvals:
    
    @staticmethod
    def get_all_scores(candidates, references):
        r1_precision = []
        r2_precision = []
        rl_precision = []
        bleu_scores = []

        r1_precision, r2_precision, rl_precision = calc_rlscore(candidates, references)
        bleu_scores = calc_bleu(candidates, references)
        bert_score_precision, bert_score_recall, bert_score_f1 = calc_bertscore(candidates, references)
        bleurt_score = calc_bleurtscore(candidates, references)
        return r1_precision, r2_precision, rl_precision, bleu_scores, bert_score_precision, bert_score_recall, bert_score_f1, bleurt_score
    
    @staticmethod
    def sft_eval(sft_path):
        model_names = os.listdir(sft_path)
        # model_names = ['mistral_16.xlsx']
        total_models = len(model_names)
        start_time = time.time()
        for idx, model_name in enumerate(model_names, start=1):
            if model_name.endswith(".xlsx") == False:
                continue
            model_excel_path = os.path.join(sft_path, model_name)
            df = pd.read_excel(model_excel_path)
            gt_answers = df["answer"].to_list()
            llm_answers = df["llm_ans"].to_list()
            r1_precision, r2_precision, rl_precision, bleu_scores, bert_score_precision, bert_score_recall, bert_score_f1, bleurt_score = RunEvals.get_all_scores(llm_answers, gt_answers)

            df["r1_precision"] = r1_precision
            df["r2_precision"] = r2_precision
            df["rl_precision"] = rl_precision
            df["bleu_score"] = bleu_scores
            df["bert_score_precision"] = bert_score_precision
            df["bert_score_recall"] = bert_score_recall
            df["bert_score_f1"] = bert_score_f1
            df["bleurt_score"] = bleurt_score
            sft_save_path = os.path.join(RESULTS_SAVE_PATH, "sft")
            if not os.path.exists(sft_save_path):
                os.mkdir(sft_save_path)
            df.to_excel(os.path.join(sft_save_path, model_name), index=False)
            
            print(f"\nModel: {model_name} Evaluation Done!")
        print("\nSFT: All Models Evaluation Done!")
        
    @staticmethod
    def rag_eval(rag_path):
        model_names = os.listdir(rag_path)
        # model_names = ['falcon_7b_instruct.xlsx']
        total_models = len(model_names)
        for idx, model_name in enumerate(model_names, start=1):
            if model_name.endswith(".xlsx") == False:
                continue
            print(f"\nModel: {model_name}")
            model_excel_path = os.path.join(rag_path, model_name)
            df = pd.read_excel(model_excel_path)
            gt_answers = df["gt"].to_list()
            llm_answers = df["llm_ans"].to_list()
            r1_precision, r2_precision, rl_precision, bleu_scores, bert_score_precision, bert_score_recall, bert_score_f1, bleurt_score = RunEvals.get_all_scores(llm_answers, gt_answers)

            df["r1_precision"] = r1_precision
            df["r2_precision"] = r2_precision
            df["rl_precision"] = rl_precision
            df["bleu_score"] = bleu_scores
            df["bert_score_precision"] = bert_score_precision
            df["bert_score_recall"] = bert_score_recall
            df["bert_score_f1"] = bert_score_f1
            df["bleurt_score"] = bleurt_score
            sft_save_path = os.path.join(RESULTS_SAVE_PATH, "rag")
            if not os.path.exists(sft_save_path):
                os.mkdir(sft_save_path)
            df.to_excel(os.path.join(sft_save_path, model_name), index=False)
            
            print(f"\nModel: {model_name} Evaluation Done!")
        print("\nSFT: All Models Evaluation Done!")
        
    @staticmethod
    def zeroshot_eval(zeroshot_path):
        model_names = os.listdir(zeroshot_path)
        done_models = os.listdir('/home/xhitij.cm/modelcards/evaluation_results/zeroshot')
        total_models = len(model_names)
        for idx, model_name in enumerate(model_names, start=1):
            if model_name.endswith(".xlsx") == False:
                continue
            if model_name in done_models:
                continue
            print(f"\nModel: {model_name}")
            model_excel_path = os.path.join(zeroshot_path, model_name)
            df = pd.read_excel(model_excel_path)
            df.fillna("", inplace=True)
            gt_answers = df["gt"].to_list()
            llm_answers = df["llm_ans"].to_list()
            r1_precision, r2_precision, rl_precision, bleu_scores, bert_score_precision, bert_score_recall, bert_score_f1, bleurt_score = RunEvals.get_all_scores(llm_answers, gt_answers)

            df["r1_precision"] = r1_precision
            df["r2_precision"] = r2_precision
            df["rl_precision"] = rl_precision
            df["bleu_score"] = bleu_scores
            df["bert_score_precision"] = bert_score_precision
            df["bert_score_recall"] = bert_score_recall
            df["bert_score_f1"] = bert_score_f1
            df["bleurt_score"] = bleurt_score
            sft_save_path = os.path.join(RESULTS_SAVE_PATH, "zeroshot")
            if not os.path.exists(sft_save_path):
                os.mkdir(sft_save_path)
            df.to_excel(os.path.join(sft_save_path, model_name), index=False)
            
            print(f"\nModel: {model_name} Evaluation Done!")
        print("\nSFT: All Models Evaluation Done!")
        
    @staticmethod
    def longcontext_eval(longcontext_path):
        model_names = os.listdir(longcontext_path)
        total_models = len(model_names)
        for idx, model_name in enumerate(model_names, start=1):
            if model_name.endswith(".xlsx") == False:
                continue

            model_excel_path = os.path.join(longcontext_path, model_name)
            df = pd.read_excel(model_excel_path)
            gt_answers = df["gt"].to_list()
            if 'llama3_8b_it_ans' in df.columns:
                llm_answers = df["llama3_8b_it_ans"].to_list()
            else:
                llm_answers = df["llm_ans"].to_list()
            r1_precision, r2_precision, rl_precision, bleu_scores, bert_score_precision, bert_score_recall, bert_score_f1, bleurt_score = RunEvals.get_all_scores(llm_answers, gt_answers)

            df["r1_precision"] = r1_precision
            df["r2_precision"] = r2_precision
            df["rl_precision"] = rl_precision
            df["bleu_score"] = bleu_scores
            df["bert_score_precision"] = bert_score_precision
            df["bert_score_recall"] = bert_score_recall
            df["bert_score_f1"] = bert_score_f1
            df["bleurt_score"] = bleurt_score
            sft_save_path = os.path.join(RESULTS_SAVE_PATH, "longcontext")
            if not os.path.exists(sft_save_path):
                os.mkdir(sft_save_path)
            df.to_excel(os.path.join(sft_save_path, model_name), index=False)
            
            print(f"\nModel: {model_name} Evaluation Done!")
        print("\nSFT: All Models Evaluation Done!")
        

if __name__ == "__main__":
    sft_results_path = "/home/xhitij.cm/modelcards/qa/sft/results"
    rag_results_path = "/home/xhitij.cm/modelcards/qa/rag/retriever/modelcard/results"
    longcontext_path = "/home/xhitij.cm/modelcards/qa/longcontext/results"
    zeroshot_path = "/home/xhitij.cm/modelcards/opendomainqa/results"

    # RunEvals.sft_eval(sft_results_path) DONE
    # RunEvals.rag_eval(rag_results_path)
    # RunEvals.zeroshot_eval(zeroshot_path)
    # RunEvals.longcontext_eval(longcontext_path) DONE
    
    print("Evaluation Done!")
