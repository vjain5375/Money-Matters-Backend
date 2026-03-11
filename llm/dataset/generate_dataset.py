import json
import random

def generate_sample():

    food = random.randint(5,40)
    shopping = random.randint(5,50)
    transport = random.randint(5,20)
    utilities = random.randint(5,30)

    spending = {
        "Food": food,
        "Shopping": shopping,
        "Transport": transport,
        "Utilities": utilities
    }

    highest = max(spending, key=spending.get)

    advice = f"You are spending too much on {highest}. Reduce it by 15% and increase savings."

    return {
        "instruction": "Give financial advice",
        "input": str(spending),
        "output": advice
    }


dataset = []

for _ in range(5000):
    dataset.append(generate_sample())

with open("finance_advice_dataset.json","w") as f:
    json.dump(dataset,f,indent=2)

print("Dataset generated")