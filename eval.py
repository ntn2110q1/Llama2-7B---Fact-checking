from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from utils.prompt import Prompter

import os

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np

def predict_label(model, tokenizer, prompter, instruction, input_text, evidence, max_new_tokens=20):

    prompt = prompter.generate_prompt(instruction, input_text, evidence)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_label = prompter.get_response(output_text)

    valid_labels = ["true", "half", "false"]
    predicted_label = predicted_label.lower()
    if predicted_label not in valid_labels:
        predicted_label = "false"
    return predicted_label

def evaluate_model(test_path, model, tokenizer, prompter):
    with open(test_path) as f:
        test_data = json.load(f)

    true_labels = []
    pred_labels = []
    instruction = "Evaluate the statement and classify it as one of the following based on the provided evidence: True, Half , False."

    for item in test_data:
        true_label = item["output"].lower()
        predicted_label = predict_label(
            model,
            tokenizer,
            prompter,
            instruction,
            item["input"],
            item["evidence"]
        )
        true_labels.append(true_label)
        pred_labels.append(predicted_label)

    acc = accuracy_score(true_labels, pred_labels)

    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_model = "yahma/llama-7b-hf"
test_path = "/content/rawfc_cleaned_test.json"
output_dir = "./lora-alpaca-rawfc"
device_map = "auto"


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


model = PeftModel.from_pretrained(model, output_dir)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# Tải template prompt
prompter = Prompter("alpaca")

# Đánh giá
results = evaluate_model(test_path, model, tokenizer, prompter)

# In kết quả
print("Evaluation results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision (macro): {results['precision_macro']:.4f}")
print(f"F1-score (macro): {results['f1_macro']:.4f}")
print(f"Precision (weighted): {results['precision_weighted']:.4f}")
print(f"F1-score (weighted): {results['f1_weighted']:.4f}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_model = "yahma/llama-7b-hf"
test_path = "/content/rawfc_cleaned_test.json"
output_dir = "./lora-alpaca-rawfc"
device_map = "auto"


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


model = PeftModel.from_pretrained(model, output_dir)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# Tải template prompt
prompter = Prompter("alpaca")

# Đánh giá
results = evaluate_model(test_path, model, tokenizer, prompter)

# In kết quả
print("Evaluation results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision (macro): {results['precision_macro']:.4f}")
print(f"F1-score (macro): {results['f1_macro']:.4f}")
print(f"Precision (weighted): {results['precision_weighted']:.4f}")
print(f"F1-score (weighted): {results['f1_weighted']:.4f}")