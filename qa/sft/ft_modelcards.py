import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

def finetune_modelcard(argmname, argrank):
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

    mname = argmname # "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" # "unsloth/llama-3-8b-bnb-4bit"
    if mname == "unsloth/mistral-7b-instruct-v0.2-bnb-4bit":
        save_dir = f"mistral_it_{argrank}" # "llama3"
    elif mname == "unsloth/llama-3-8b-bnb-4bit":
        save_dir = f"llama3_{argrank}"
    elif mname == "unsloth/mistral-7b-bnb-4bit":
        save_dir = f"mistral_{argrank}"
    elif mname == "unsloth/gemma-2b-it-bnb-4bit":
        save_dir = f"gemma_{argrank}"
    else:
        dirname = argmname.replace("/", "_")
        dirname = dirname.replace("-", "_")
        save_dir = dirname + f"_{argrank}"
    if not os.path.exists(f"./sft_models/{save_dir}"):
        os.makedirs(f"./sft_models/{save_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = mname,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = argrank, #8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        # instructions = examples["instruction"]
        instructions = ["You are provided with an excerpt from a research paper about a language model in the domain of NLP and DL. You are provided with a question about the language model. Answer the question based on the provided text and your knowledge of NLP."]*len(examples["answer"])
        inputs = []
        for title, context, question in zip(examples["title"], examples["context"], examples["question"]):
            input_text = "Paper Title - " + title + "\nPaper Excerpt - " + " ".join(context[0:3]) + "\n\nQuestion: " + question
            inputs.append(input_text)
        
        outputs      = examples["answer"]
        
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    dataset = load_dataset('json', data_files="../../data/modelcard_ft_data/train.jsonl")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    train_valid_dataset = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = train_valid_dataset['train']
    val_dataset   = train_valid_dataset['test']

    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset   = val_dataset.shuffle(seed=42)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.03)],
        packing = False, # Can make training 5x faster for short sequences.

        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs=10,
            evaluation_strategy='epoch',
            max_steps = 0,
            learning_rate = 2e-4,
            fp16 = True, #not torch.cuda.is_bf16_supported(),
            bf16 = False, #torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_total_limit = 2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        ),
    )

    trainer_stats = trainer.train()

    trainer.save_model(f"./sft_models/{save_dir}")

    print("Model training done, model saved. Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='name of the model to use for generation, specifically the hf repo name')
    parser.add_argument('--rank', help='rank of matrix')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    finetune_modelcard(args.model_name, int(args.rank))