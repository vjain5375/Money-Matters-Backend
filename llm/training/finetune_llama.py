import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

print("Adding LoRA adapters...")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="finance_advice_dataset.json"
)

training_args = TrainingArguments(
    output_dir="finance_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="output",
    args=training_args,
)

print("Training started...")

trainer.train()

trainer.model.save_pretrained("finance_llama")
tokenizer.save_pretrained("finance_llama")

print("Model saved successfully")