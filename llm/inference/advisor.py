import os
from dotenv import load_dotenv
load_dotenv()

# Login
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit" 
adapter_path = "MoneyMattersAI/models/finance_llama" 

print("Loading base model into GPU...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map={"": 0})

print("Applying finance adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

def generate_advice(spending_dict):
    # Llama-3 Instruct Format for better control
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Analyze this spending pattern: {spending_dict}. "
        f"Give ONE short, unique financial advice paragraph. Do not repeat.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nAdvice:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating AI Advice...\n")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=80,                # Space kam kar di taaki bakwas na kare
        temperature=0.5,                  # Thoda serious tone
        top_p=0.9,                        
        repetition_penalty=1.3,           # Loop rokne ke liye penalty badha di
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id 
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's part after "Advice:"
    if "Advice:" in full_output:
        # split("\n")[0] se sirf pehla complete paragraph milega
        clean_advice = full_output.split("Advice:")[-1].split("\n")[0].strip()
        return clean_advice
    return full_output

test_spending = {"Food": 20, "Shopping": 45, "Transport": 10, "Utilities": 25}

print("\n--- AI FINANCIAL ADVISOR OUTPUT ---")
print(generate_advice(test_spending))