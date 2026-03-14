import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Finance LLaMA Advisor API")

# -- Paths --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "finance_llama")
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# -- Lazy load model (loaded once on startup) --
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        print("Loading base model...")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto")
        print(f"Applying LoRA adapters from: {ADAPTER_PATH}")
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()
        print("Model ready!")
    return _model, _tokenizer


class SpendingData(BaseModel):
    expenses: dict


@app.post("/get-advice")
async def get_advice(data: SpendingData):
    try:
        model, tokenizer = get_model()
        spending_dict = data.expenses

        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Analyze this spending pattern: {spending_dict}. "
            f"Give ONE short, unique financial advice paragraph. Do not repeat.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nAdvice:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        clean_advice = full_output.split("Advice:")[-1].split("\n")[0].strip() \
            if "Advice:" in full_output else full_output.strip()

        return {"advice": clean_advice}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)