
import evaluate
from evaluate import load
from rouge_score import rouge_scorer
import gc


def calc_rlscore(gen_ans_array, ans_array):
    rscorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def get_one(gen_ans, ans):
        ans = str(ans).strip()
        gen_ans = str(gen_ans).strip()
        scores = rscorer.score(ans, gen_ans)
        return scores['rouge1'].precision*100, scores['rouge2'].precision*100, scores['rougeL'].precision*100
    
    results = [get_one(gen_ans, ans) for gen_ans, ans in zip(gen_ans_array, ans_array)]
    
    # Flush the model
    del rscorer
    gc.collect()
    
    return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results]
    

def calc_bleu(predictions_array, references_array):
    bleu = evaluate.load("bleu")
    
    def get_one(predictions, references):
        try:
            results = bleu.compute(predictions=predictions, references=references)
            return results['bleu']
        except Exception as ex:
            print("Error encountered in bleu computation: ", ex)
            print("Preds: ", predictions)
            print("Refs: ", references)
            return 0
        
    results = [get_one([predictions], [references]) for predictions, references in zip(predictions_array, references_array)]
    
    # Flush the model
    del bleu
    gc.collect()
    
    return results

def calc_bertscore(predictions, references):
    bertscore = load("bertscore")
    # return [-1 for _ in range(len(predictions))], [-1 for _ in range(len(predictions))], [-1 for _ in range(len(predictions))]
    results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
    precision = [score*100 for score in results['precision']]
    recall = [score*100 for score in results['recall']]
    f1 = [score*100 for score in results['f1']]

    # try:
    #     results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
    #     precision = [score*100 for score in results['precision']]
    #     recall = [score*100 for score in results['recall']]
    #     f1 = [score*100 for score in results['f1']]
    # except Exception as ex:
    #     import pdb; pdb.set_trace()
    #     print("Error during bertscore computation: ", ex)
    #     return None
    
    # Flush the model
    del bertscore
    gc.collect()
    
    return precision, recall, f1

def calc_bleurtscore(predictions, references):
    return 0
    # bleurt = load("bleurt", module_type="metric")
    # results = bleurt.compute(predictions=predictions, references=references)
    # scores = [score*100 for score in results["scores"]]
    
    # # Flush the model
    # del bleurt
    # gc.collect()
    
    # return scores
