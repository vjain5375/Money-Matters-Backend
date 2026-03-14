import os
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()

# -- Resolve paths --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "finance_llama")

# Login to HuggingFace
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# unsloth/llama-3-8b-Instruct-bnb-4bit is ALREADY 4-bit quantized
# Do NOT pass BitsAndBytesConfig — it conflicts with the model's built-in quant config
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",   # automatically puts model on GPU
)
print(f"✅ Loaded! Device: {next(base_model.parameters()).device}")

print(f"Applying LoRA adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("✅ Model ready!")




def generate_advice(spending_dict: dict) -> str:
    """
    Generate financial advice given a spending breakdown dict.
    
    Args:
        spending_dict: e.g. {"Food": 3500, "Shopping": 8000, "Transport": 1200}
    
    Returns:
        AI-generated financial advice string.
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Analyze this spending pattern: {spending_dict}. "
        f"Give ONE short, unique financial advice paragraph. Do not repeat.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nAdvice:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating AI advice...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's answer after "Advice:"
    if "Advice:" in full_output:
        clean_advice = full_output.split("Advice:")[-1].split("\n")[0].strip()
        return clean_advice
    return full_output.strip()


# -- Quick test when run directly --
if __name__ == "__main__":
    test_spending = {"Food": 3500, "Shopping": 8000, "Transport": 1200, "Utilities": 2000}
    print("\n--- AI FINANCIAL ADVISOR OUTPUT ---")
    print(generate_advice(test_spending))