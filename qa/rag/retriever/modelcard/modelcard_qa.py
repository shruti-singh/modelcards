import json
import torch
from vllm import LLM, SamplingParams
import argparse
import pandas as pd

model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat", "meta-llama/Llama-2-7b-hf": "llama2_7b", "meta-llama/Llama-2-13b-hf": "llama2_13b", "meta-llama/Llama-2-70b-hf": "llama2_70b", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b_instruct_v2", "mistralai/Mixtral-8x7B-v0.1": "mistral_8_7b", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral_8_7b_instruct", "lmsys/vicuna-13b-v1.5": "vicuna_13b", "lmsys/vicuna-13b-v1.5-16k": "vicuna_13b_16k", "lmsys/longchat-7b-v1.5-32k": "longchat_7b_32k", "lmsys/vicuna-7b-v1.5": "vicuna_7b", "lmsys/vicuna-7b-v1.5-16k": "vicuna_7b_16k", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "tiiuae/falcon-40b": "falcon_40b", "facebook/galactica-6.7b": "galactica_7b", "facebook/galactica-30b": "galactica_30b", "microsoft/phi-2": "ms_phi2_3b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "Qwen/Qwen2-beta-7B-Chat": "qwen_7b_chat", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b_it", "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3_81b_it", "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama3_70b_it"}

model2contextlength = {
    "meta-llama/Llama-2-7b-chat-hf": ("llama2_7b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json"),
    "mistralai/Mistral-7B-v0.1": ("mistral_7b", 8000, "https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/#:~:text=Mistral%207B%20has%20an%208%2C000,at%20a%207B%20model%20size."),
    "mistralai/Mistral-7B-Instruct-v0.1": ("mistral_7b_instruct", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"),
    "tiiuae/falcon-7b": ("falcon_7b", 2048, "https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer_config.json"),
    "tiiuae/falcon-7b-instruct": ("falcon_7b_instruct", 2048, "https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/tokenizer_config.json"), 
    "facebook/galactica-6.7b": ("galactica_7b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "facebook/galactica-30b": ("galactica_30b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "tiiuae/falcon-40b": ("falcon_40b", 2048, "https://huggingface.co/tiiuae/falcon-40b/blob/main/tokenizer_config.json"),
    "google/gemma-2b-it": ("gemma_2b_it", 8192, ""), 
    "google/gemma-2b": ("gemma_2b", 8192, ""), 
    "meta-llama/Meta-Llama-3-8B-Instruct": ("llama3_8b_it", 8192, ""),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ("llama3_81b_it", 8192, "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B"),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ("llama3_70b_it", 8192, "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B")
}

def model_response_gen(model_name):
    num_cuda = torch.cuda.device_count()
    mname = model_name_map[model_name]

    if model_name in ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype="half", trust_remote_code=True)
    elif model_name in  ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype="half", trust_remote_code=True)
    elif model_name in ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b"]: 
        llm = LLM(model=model_name, tensor_parallel_size=1, dtype="half")
    elif model_name in ["google/gemma-2b-it", "google/gemma-2b"]:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, dtype=torch.float32)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=num_cuda, trust_remote_code=True)
    
    model_tokenizer = llm.get_tokenizer()

    # This is input+output tokens, so we reserve 400 tokens for output generation.
    model_max_len = model2contextlength[model_name][1] - 400 #llm.llm_engine.model_config.max_model_len
    clean_model_name = model_name_map[model_name]

    model_ans_list = []
    prompts_list = []
    
    # Read the qa data
    with open("../../../modelcard_qa_topk.json", "r") as f:
        qa_data = json.load(f)

    for _, dockey in enumerate(qa_data):

        ldb_gen_prompt = "You are provided with a question about a research paper in NLP and DL. You are also provided with an excerpt from the paper above that contains the answer. Answer the question based on the excerpt. Do not include any additional text in the output except the answer to the question.\n" + qa_data[dockey]["question"] + "\nAnswer: "
        
        prompt_len = len(model_tokenizer.tokenize(ldb_gen_prompt))
        topk_paras_text = "Excerpt: "
        topk_paras_text += qa_data[dockey]["topk_data"]
        topk_paras_text += "\n"
        
        context_tokens_list = model_tokenizer.tokenize(topk_paras_text)
        if (len(context_tokens_list) + prompt_len) < model_max_len:
            ldb_gen_prompt = topk_paras_text + ldb_gen_prompt
        else:
            croppped_context = model_tokenizer.convert_tokens_to_string(context_tokens_list[:(model_max_len-prompt_len-3)])
            ldb_gen_prompt = croppped_context + ldb_gen_prompt

        prompts_list.append(ldb_gen_prompt)
        pkey = dockey.rsplit("/", 1)[-1].replace(".json", "")
        local_dict = {'prompt': ldb_gen_prompt, 'question': qa_data[dockey]["question"], 'gt': qa_data[dockey]["answer"], 'pkey': pkey}
        model_ans_list.append(local_dict)
    
    print("Collated all model card generation prompts, now generating outputs...")
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=1600, stop=['Question:', '\n\n\n\n', '| --- | --- | --- | --- | --- | ', '| | | |'])
    # To generate in batches and save answers every 500 steps
    prompt_list_chunks = [prompts_list[x:x+500] for x in range(0, len(prompts_list), 500)]

    generated_ans = []
    for prmpt_chunk in prompt_list_chunks:
        outputs = llm.generate(prmpt_chunk, sampling_params)
        for output in outputs:
            generated_ans.append(output.outputs[0].text)

        for _, ans in enumerate(generated_ans):
            model_ans_list[_][f'llm_ans'] = ans

        df = pd.DataFrame(model_ans_list)
        # df.to_excel(f'./llm_results/may27/{clean_model_name}.xlsx')
        df.to_excel(f'./outputs/{clean_model_name}.xlsx')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='name of the model to use for generation, specifically the hf repo name')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model_response_gen(args.model_name)