import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import load_dataset

def infer_results(argmname, argrank):
    max_seq_length = 2048
    dtype = torch.float16 #None
    load_in_4bit = True

    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
        "unsloth/gemma-7b-it-bnb-4bit",
        "unsloth/gemma-2b-bnb-4bit",
        "unsloth/gemma-2b-it-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",
    ]
    mname = argmname
    if mname == "unsloth/mistral-7b-instruct-v0.2-bnb-4bit":
        save_dir = f"mistral_it_{argrank}" # "llama3"
    elif mname == "unsloth/llama-3-8b-bnb-4bit":
        save_dir = f"llama3_{argrank}"
    elif mname == "unsloth/mistral-7b-bnb-4bit":
        save_dir = f"mistral_{argrank}"
    else:
        dirname = argmname.replace("/", "_")
        dirname = dirname.replace("-", "_")
        save_dir = dirname + f"_{argrank}"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"./sft_models/{save_dir}",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)


    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    dataset = load_dataset('json', data_files="../../data/modelcard_ft_data/test.jsonl")

    llm_ans_list = []
    for _, data in enumerate(dataset["train"]):
        context_str = " ".join(data['context'])
        context_str = context_str.replace("\n", " ")
        inputs = tokenizer(
        [
        f"You are provided with an excerpt from a research paper about a language model in the domain of NLP and DL. You are provided with a question about the language model. Answer the question based on the provided text and your knowledge of NLP.\n\nPaper Title - {data['title']} Paper Excerpt: {context_str}\n\nQuestion: {data['question']}\n\n### Response:" #+ EOS_TOKEN padding_side='left',
        ],  return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True, pad_token_id=tokenizer.eos_token_id)
        gen_answer = tokenizer.batch_decode(outputs)
        
        local_ans_dict = data
        local_ans_dict['llm_ans'] = gen_answer[0].split("### Response:")[1]
        llm_ans_list.append(local_ans_dict)
        if _ % 10 == 0:
            print(_)
            df = pd.DataFrame.from_records(llm_ans_list)
            df.to_excel(f"./results/{save_dir}.xlsx")

    df = pd.DataFrame.from_records(llm_ans_list)
    df.to_excel(f"./results/{save_dir}.xlsx")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='name of the model to use for generation, specifically the hf repo name')
    parser.add_argument('--rank', help='rank of matrix')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    infer_results(args.model_name, int(args.rank))