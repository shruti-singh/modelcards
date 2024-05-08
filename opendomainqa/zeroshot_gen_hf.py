import pandas as pd
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_text(prompt_text):
    global llm_model, llm_tokenizer
    messages = [
    {"role": "system", "content": "You are provided with a question about a languge model in natural language processing. Based on your knowledge of deep learning and natural language processing, answer the question. Also provide evidence from the paper that supports the answer."},
    {"role": "user", "content": prompt_text},
    ]
    input_ids = llm_tokenizer.apply_chat_template(messages,
    add_generation_prompt=True,
    return_tensors="pt").to(llm_model.device)

    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        pad_token_id=llm_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    answer = llm_tokenizer.decode(response, skip_special_tokens=True)
    return answer



model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat", "meta-llama/Llama-2-7b-hf": "llama2_7b", "meta-llama/Llama-2-13b-hf": "llama2_13b", "meta-llama/Llama-2-70b-hf": "llama2_70b", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b_instruct_v2", "mistralai/Mixtral-8x7B-v0.1": "mistral_8_7b", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral_8_7b_instruct", "lmsys/vicuna-13b-v1.5": "vicuna_13b", "lmsys/vicuna-13b-v1.5-16k": "vicuna_13b_16k", "lmsys/longchat-7b-v1.5-32k": "longchat_7b_32k", "lmsys/vicuna-7b-v1.5": "vicuna_7b", "lmsys/vicuna-7b-v1.5-16k": "vicuna_7b_16k", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "tiiuae/falcon-40b": "falcon_40b", "facebook/galactica-6.7b": "galactica_7b", "facebook/galactica-30b": "galactica_30b", "microsoft/phi-2": "ms_phi2_3b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "Qwen/Qwen2-beta-7B-Chat": "qwen_7b_chat", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b_it"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=list(model_name_map.keys()), default="meta-llama/Meta-Llama-3-8B-Instruct", help='name of the model to use for generation, specifically the hf repo name')
    args = parser.parse_args()
    return args

def generate_modelcard(model_name):
    mname = model_name_map[model_name]
    sheet_to_df_map = pd.read_excel('../data/QAData.xlsx', sheet_name=None)
    with pd.ExcelWriter('../data/output/{}.xlsx') as writer:
        for _, mod_key in enumerate(sheet_to_df_map.keys()):
            print(f"Processing model {_}: {mod_key}")
            model_df = sheet_to_df_map[mod_key].loc[9:]
            col_name = sheet_to_df_map[mod_key].columns[1]
            model_dict_list = []

            for i in model_df.iterrows():
                prompt_text = i[1][col_name]
                ans = get_text(prompt_text)
                local_dict = {'prompt': prompt_text, f'{mname}_answer': ans}
                model_dict_list.append(local_dict)

            subdf = pd.DataFrame(model_dict_list)
            subdf.to_excel(writer, sheet_name=mod_key)


if __name__ == "__main__":
    args = parse_args()
    generate_modelcard(args.model_name)
