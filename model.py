import torch
import transformers

import fire
import os
import sys

from datasets import load_dataset

from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    TaskType
)

import warnings
import fire

from utils.prompt import Prompter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings("ignore")

def train(
    base_model: str = "yahma/llama-7b-hf",
    train_path: str = "/content/liarraw_cleaned_train.json",
    val_path: str = "/content/liarraw_cleaned_val.json",
    output_dir: str = "./lora-alpaca-rawfc",
    batch_size: int = 32,    #liar: 32; rawfc: 16
    micro_batch_size: int = 8,
    num_epochs: int = 3,    #liar: 3; rawfc: 1
    learning_rate: float = 5e-5,
    cutoff_len: int = 512,
    val_set_size: int = 0,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"],
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_path: {train_path}\n"
            f"val_path: {val_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='yahma/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Cấu hình lượng tử hóa 8-bit
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["evidence"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"], data_point["evidence"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if train_path.endswith(".json") or train_path.endswith(".jsonl"):
        data = load_dataset("json", data_files={"train": train_path, "val": val_path})
    else:
        data = load_dataset(data_files={"train": train_path, "val": val_path})

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = data["val"].shuffle().map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=20,
            save_steps=20,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    with torch.no_grad():
        model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    fire.Fire(train)