from vllm import LLM, SamplingParams
import pandas as pd
import torch
import argparse

model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat", "meta-llama/Llama-2-7b-hf": "llama2_7b", "meta-llama/Llama-2-13b-hf": "llama2_13b", "meta-llama/Llama-2-70b-hf": "llama2_70b", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b_instruct_v2", "mistralai/Mixtral-8x7B-v0.1": "mistral_8_7b", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral_8_7b_instruct", "lmsys/vicuna-13b-v1.5": "vicuna_13b", "lmsys/vicuna-13b-v1.5-16k": "vicuna_13b_16k", "lmsys/longchat-7b-v1.5-32k": "longchat_7b_32k", "lmsys/vicuna-7b-v1.5": "vicuna_7b", "lmsys/vicuna-7b-v1.5-16k": "vicuna_7b_16k", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "tiiuae/falcon-40b": "falcon_40b", "facebook/galactica-6.7b": "galactica_7b", "facebook/galactica-30b": "galactica_30b", "microsoft/phi-2": "ms_phi2_3b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "Qwen/Qwen2-beta-7B-Chat": "qwen_7b_chat", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b"}

model2contextlength = {
    "Qwen/Qwen2-beta-7B-Chat": ("qwen_7b_chat", 8192, "Qwen/Qwen2-beta-7B-Chat"),
    "meta-llama/Llama-2-7b-chat-hf": ("llama2_7b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-chat-hf": ("llama2_13b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/blob/main/generation_config.json"), 
    "meta-llama/Llama-2-70b-chat-hf": ("llama2_70b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-7b-hf": ("llama2_7b", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-hf": ("llama2_13b", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-70b-hf": ("llama2_70b", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/generation_config.json"),
    "mistralai/Mistral-7B-v0.1": ("mistral_7b", 8000, "https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/#:~:text=Mistral%207B%20has%20an%208%2C000,at%20a%207B%20model%20size."),
    "mistralai/Mistral-7B-Instruct-v0.1": ("mistral_7b_instruct", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"),
    # "mistralai/Mistral-7B-Instruct-v0.2": ("mistral_7b_instruct_v2", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"), 
    "mistralai/Mixtral-8x7B-v0.1": ("mistral_8_7b", 32000, "https://mistral.ai/news/mixtral-of-experts/"), 
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("mistral_8_7b_instruct", 32000, "https://mistral.ai/news/mixtral-of-experts/"),
    "lmsys/vicuna-13b-v1.5": ("vicuna_13b", 4096, "https://huggingface.co/lmsys/vicuna-13b-v1.5/blob/main/generation_config.json"),
    "lmsys/vicuna-13b-v1.5-16k": ("vicuna_13b_16k", 16384, "https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/generation_config.json"), 
    "lmsys/longchat-7b-v1.5-32k": ("longchat_7b_32k", 32768, "https://huggingface.co/lmsys/longchat-7b-v1.5-32k/blob/main/tokenizer_config.json"),
    "lmsys/vicuna-7b-v1.5": ("vicuna_7b", 4096, "https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/generation_config.json"),
    "lmsys/vicuna-7b-v1.5-16k": ("vicuna_7b_16k", 16384, "https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/blob/main/generation_config.json"), 
    "HuggingFaceH4/zephyr-7b-beta": ("zephyr_7b_beta", 16384, "https://docs.endpoints.anyscale.com/supported-models/huggingfaceh4-zephyr-7b-beta/"),
    "tiiuae/falcon-7b": ("falcon_7b", 2048, "https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer_config.json"),
    "tiiuae/falcon-7b-instruct": ("falcon_7b_instruct", 2048, "https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/tokenizer_config.json"), 
    "facebook/galactica-6.7b": ("galactica_7b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "facebook/galactica-30b": ("galactica_30b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "microsoft/phi-2": ("ms_phi2_3b", 2048, "https://huggingface.co/microsoft/phi-2"),
    "tiiuae/falcon-40b": ("falcon_40b", 2048, "https://huggingface.co/tiiuae/falcon-40b/blob/main/tokenizer_config.json"),
    "google/gemma-2b-it": ("gemma_2b_it", 8192, ""), 
    "google/gemma-2b": ("gemma_2b", 8192, ""), 
    "meta-llama/Meta-Llama-3-8B-Instruct": ("llama3_8b", 4096, "")
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=list(model_name_map.keys()), help='name of the model to use for generation, specifically the hf repo name')
    args = parser.parse_args()
    return args

def generate_modelcard(model_name):
    num_cuda = 1 # torch.cuda.device_count()
    mname = model_name_map[model_name]
    
    # For models like mistral, falcon, cc>=8.0 required, so we load in 
    if model_name in ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1",  "HuggingFaceH4/zephyr-7b-beta"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype="half", trust_remote_code=True)
    elif model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype=torch.bfloat16, trust_remote_code=True)
    elif model_name in ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b"]: 
        llm = LLM(model=model_name, tensor_parallel_size=1, dtype="half")
    elif model_name in ["microsoft/phi-2"]:
        llm = LLM("microsoft/phi-2", download_dir="./data/vllmmodels/", trust_remote_code=True)
        #llm = PhiForCausalLM(model_name="microsoft/phi-2")
    elif model_name in ["google/gemma-2b-it", "google/gemma-2b"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype=torch.float32)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, trust_remote_code=True)

    temp = 0.1
    topp = 0.9

    sheet_to_df_map = pd.read_excel('../data/QAData.xlsx', sheet_name=None)
    with pd.ExcelWriter('../data/output/{}.xlsx') as writer:
        all_prompts = []
        for _, mod_key in enumerate(sheet_to_df_map.keys()):
            print(f"Processing model {_}: {mod_key}")
            model_df = sheet_to_df_map[mod_key].loc[9:]
            col_name = sheet_to_df_map[mod_key].columns[1]
            model_dict_list = []
            sampling_params = SamplingParams(temperature=temp, top_p=topp, max_tokens=512)

            for i in model_df.iterrows():
                prompt_text = i[1][col_name]
                all_prompts.append(prompt_text)
            
            outputs = llm.generate(all_prompts, sampling_params)
            for output in outputs:
                local_dict = {'prompt': prompt_text, f'{mname}_answer': output.outputs[0].text}
                model_dict_list.append(local_dict)
                
            subdf = pd.DataFrame(model_dict_list)
            subdf.to_excel(writer, sheet_name=mod_key)


if __name__ == "__main__":
    args = parse_args()
    generate_modelcard(args.model_name)