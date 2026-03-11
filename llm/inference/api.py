from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

# Configuration
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
adapter_path = "MoneyMattersAI/models/finance_llama"

print("🚀 Loading Model into GPU... Please wait.")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map={"": 0})
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ Model Ready!")

# Request Body Structure
class SpendingData(BaseModel):
    expenses: dict

@app.post("/get-advice")
async def get_advice(data: SpendingData):
    try:
        spending_dict = data.expenses
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Analyze this spending pattern: {spending_dict}. "
            f"Give ONE short, unique financial advice paragraph. Do not repeat.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nAdvice:"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=80,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id 
        )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_advice = full_output.split("Advice:")[-1].split("\n")[0].strip()
        
        return {"advice": clean_advice}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)