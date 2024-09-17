from vllm import LLM, SamplingParams
import pandas as pd
import torch
import argparse

model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "facebook/galactica-6.7b": "galactica_7b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b_it", "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3_81b_it", "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3_70b_it"}

model2contextlength = {
    "meta-llama/Llama-2-7b-hf": ("llama2_7b", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-hf": ("llama2_13b", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/generation_config.json"),
    "mistralai/Mistral-7B-Instruct-v0.1": ("mistral_7b_instruct", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"),
    "HuggingFaceH4/zephyr-7b-beta": ("zephyr_7b_beta", 16384, "https://docs.endpoints.anyscale.com/supported-models/huggingfaceh4-zephyr-7b-beta/"),
    "tiiuae/falcon-7b": ("falcon_7b", 2048, "https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer_config.json"),
    "tiiuae/falcon-7b-instruct": ("falcon_7b_instruct", 2048, "https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/tokenizer_config.json"), 
    "facebook/galactica-6.7b": ("galactica_7b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "google/gemma-2b-it": ("gemma_2b_it", 8192, ""), 
    "google/gemma-2b": ("gemma_2b", 8192, ""), 
    "meta-llama/Meta-Llama-3-8B-Instruct": ("llama3_8b_it", 4096, ""),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ("llama3_81b_it", 4096, "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B"),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ("llama3_70b_it", 4096, "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B")
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=list(model_name_map.keys()), help='name of the model to use for generation, specifically the hf repo name')
    parser.add_argument('--config', choices=["ans_with_evidence", "ans"], help='name of the config to use for generation')
    args = parser.parse_args()
    return args

def generate_modelcard(model_name, config):
    num_cuda = torch.cuda.device_count()
    mname = model_name_map[model_name]
    
    # For models like mistral, falcon, cc>=8.0 required, so we load in 
    if model_name in ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1",  "HuggingFaceH4/zephyr-7b-beta"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype="half", trust_remote_code=True)
    elif model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype="half", trust_remote_code=True)
    elif model_name in ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"]: 
        llm = LLM(model=model_name, tensor_parallel_size=1, dtype="half")
    elif model_name in ["google/gemma-2b-it", "google/gemma-2b"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype=torch.float32)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, trust_remote_code=True)

    temp = 0.1
    topp = 0.9

    if config == "ans_with_evidence":
        output_dir = "./outputs/ans_with_evidence"
    elif config == "ans":
        output_dir = "./outputs/ans"

    sheet_to_df_map = pd.read_excel('../../data/QAData.xlsx', sheet_name=None)
    with pd.ExcelWriter(f'{output_dir}/{mname}.xlsx') as writer:
        for _, mod_key in enumerate(sheet_to_df_map.keys()):
            all_prompts = []
            model_dict_list = []
            print(f"Processing model {_}: {mod_key}")
            model_df = sheet_to_df_map[mod_key].loc[9:]
            col_name = sheet_to_df_map[mod_key].columns[1]
            ptitle = list(model_df.columns)[1].strip().replace("\n", "")
            if ptitle == "BART":
                ptitle = "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
            
            sampling_params = SamplingParams(temperature=temp, top_p=topp, max_tokens=1600, stop=['Question:', '\n\n\n\n', '| --- | --- | --- | --- | --- | ', '| | | |'])

            for i in model_df.iterrows():
                instr = "You are provided with a question about a languge model in the domain of natural language processing. Based on your knowledge of deep learning and natural language processing, answer the question."
                old_prompt = i[1][col_name]
                que = old_prompt.split(" Question: ")[1]
                if config == "ans_with_evidence":
                    # throw NotImplementedError("Not implemented config")
                    instr = instr + " Also provide evidence of the answer from the paper."
                    new_prompt = instr + " " + que
                if config == "ans":
                    new_prompt = instr + " \n" + f"Paper Title: {ptitle}" + "\nQuestion: " + que + "\n Answer:"
                all_prompts.append(new_prompt)
            
            outputs = llm.generate(all_prompts, sampling_params)
            for new_prompt, output in zip(all_prompts, outputs):
                local_dict = {'prompt': new_prompt, f'{mname}_ans': output.outputs[0].text}
                model_dict_list.append(local_dict)
                
            subdf = pd.DataFrame(model_dict_list)
            subdf.to_excel(writer, sheet_name=mod_key)


if __name__ == "__main__":
    args = parse_args()
    generate_modelcard(args.model_name, args.config)
