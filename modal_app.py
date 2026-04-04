import modal
import json
import os

# Define the Modal App
app = modal.App("finance-llama-api")

# Define the Image (Environment)
# We need PyTorch, Transformers, Peft, and BitsAndBytes for 4-bit inference
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "fastapi[standard]", 
    )
)

# Set constants
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTER_PATH = "vanshh404/finance-llama-lora"


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
class FinanceAdvisor:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading tokenizer and base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto"
        )
        self.model.eval()

        self.PROMPT_TEMPLATE = """You are a sharp, no-nonsense personal finance coach for Indians. Give 3 brutally specific tips based on the user's ACTUAL purchase history. Reference their real items. No percentages. No generic lines like "reduce spending" or "increase savings".

Example of BAD advice: "You spend too much on shopping. Reduce it by 15%."
Example of GOOD advice: "You bought a smartphone and laptop in the same month — stagger big electronics purchases across quarters so one doesn't wipe your monthly budget."

---
User's data:
Expenses this month: {spending_summary}
Recent purchases: {recent_items}{income_line}{comparison_line}

3 specific tips (reference their actual items, be direct):
1.
2.
3."""

    @modal.fastapi_endpoint(method="POST", docs=True)
    def get_advice(self, request: dict = None):
        import torch
        import re
        
        data = request if request else {}

        spending_dict = data.get("expenses", {})
        recent_items_dict = data.get("recent_items", {})
        income = data.get("income", 0)
        last_month_spend = data.get("last_month_spend", 0)
        txn_count = data.get("txn_count", 0)
        
        if not spending_dict:
            return {"advice": "No spending data found this month. Start tracking your expenses to get personalized advice!"}

        total = sum(spending_dict.values())
        if total == 0:
            return {"advice": "Your total spending is ₹0 this month. Great start — keep tracking!"}

        PRETTY = {
            'shopping': 'Shopping', 'food': 'Food & Dining', 'transport': 'Transport',
            'utilities': 'Utilities', 'entertainment': 'Entertainment', 
            'subscriptions': 'Subscriptions', 'health': 'Health & Medical',
            'salary': 'Salary', 'other_income': 'Other Income', 'other': 'Other'
        }

        sorted_cats = sorted(spending_dict.items(), key=lambda x: x[1], reverse=True)

        spending_summary = ", ".join([
            f"{PRETTY.get(cat, cat)}: ₹{amt:,.0f}" for cat, amt in sorted_cats
        ])

        recent_items_str = ", ".join([
            f"{PRETTY.get(cat, cat)} ({', '.join(items)})" 
            for cat, items in recent_items_dict.items() if items
        ]) or "None provided"

        income_line = f"\nMonthly income: ₹{income:,.0f}" if income > 0 else ""

        comparison_line = ""
        if last_month_spend > 0:
            diff = total - last_month_spend
            direction = "more" if diff > 0 else "less"
            comparison_line = f"\nThis is ₹{abs(diff):,.0f} {direction} than last month (₹{last_month_spend:,.0f})."

        prompt = self.PROMPT_TEMPLATE.format(
            spending_summary=spending_summary,
            recent_items=recent_items_str,
            total=f"{total:,.0f}",
            income_line=income_line,
            comparison_line=comparison_line,
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(input_ids, "get"):
                input_ids = input_ids.get("input_ids")
            input_ids = input_ids.to(self.model.device)

            print(f"Generating insights for {len(spending_dict)} categories, total ₹{total}, income ₹{income}...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=250,
                    temperature=0.8,
                    top_p=0.92,
                    top_k=60,
                    repetition_penalty=1.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response_ids = output_ids[0][input_ids.shape[-1]:]
            raw_output = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            print(f"Raw model output: {raw_output}")

            insights = self._parse_insights(raw_output, sorted_cats, PRETTY, total, income)
            return {"insights": insights}

        except Exception as e:
            print(f"LLaMA generation failed: {e}")
            top_name = PRETTY.get(sorted_cats[0][0], sorted_cats[0][0])
            return {"insights": [
                f"Your {top_name} spending dominates this month — review if these purchases were planned or impulsive.",
                "Try the 50/30/20 rule — allocate 50% to needs, 30% to wants, and save at least 20% of your income.",
                "Set a weekly spending cap for your top category to build better habits over time.",
            ]}

    def _parse_insights(self, raw_output, sorted_cats, pretty_map, total, income):
        import re

        # Strip any preamble before the numbered list
        raw_output = re.sub(r'^.*?(?=\n?1\.)', '', raw_output, flags=re.DOTALL).strip()
        if not raw_output:
            raw_output = self._last_raw  if hasattr(self, '_last_raw') else ''

        # Extract numbered points: "1. ...", "2. ...", "3. ..."
        points = re.findall(r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)', raw_output, re.DOTALL)
        insights = []
        for p in points[:3]:
            # Take only first sentence/line, strip markdown asterisks
            clean = p.strip().split('\n')[0].strip()
            clean = re.sub(r'\*+', '', clean).strip()
            if len(clean) > 15:
                insights.append(clean)

        # Fallback: split by newlines and take non-empty lines
        if len(insights) < 3:
            lines = [re.sub(r'^\d+\.\s*|\*+', '', l).strip() for l in raw_output.split('\n') if len(l.strip()) > 15]
            insights = list(dict.fromkeys(lines))[:3]  # deduplicate, take first 3

        # Hardcoded fallbacks if model completely failed
        top_name = pretty_map.get(sorted_cats[0][0], sorted_cats[0][0])
        fallbacks = [
            f"Your {top_name} spending is concentrated in a few big purchases — space them out across months to avoid cash crunches.",
            f"{'You are spending more than your income this month — pause non-essential purchases until next month.' if income > 0 and total > income else 'Set a weekly spending limit for your top category and track it every Sunday.'}",
            f"For every large purchase, wait 48 hours before buying — this alone reduces impulse spending significantly.",
        ]
        while len(insights) < 3:
            insights.append(fallbacks[len(insights)])

        return insights[:3]

# Local entrypoint to test
@app.local_entrypoint()
def test_local():
    advisor = FinanceAdvisor()
    test_data = {"expenses": {"Coffee": 50, "Rent": 1500, "Groceries": 300}}
    print(advisor.get_advice.local(test_data))
