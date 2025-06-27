import pandas as pd
import csv
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
from sklearn.metrics import precision_score, recall_score, f1_score

from fingpt_loader import load_fingpt, call_fingpt
from self_rag_validator import validate_response

# Load model and data
fingpt = load_fingpt()
df = pd.read_csv("merged_output.csv")
true_labels = df["Label"].tolist()
pred_labels = []

# Output log file
log_file = "scored_outputs.csv"
with open(log_file, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["ID", "Timestamp", "Prompt", "Response", "Score", "Leak", "Compliance", "Explanation"])

# Evaluate each prompt
for prompt in tqdm(df["Prompt"], desc="Processing prompts"):
    response = call_fingpt(fingpt, prompt)
    leak, compliant, score, explanation = validate_response(prompt, response)

    verdict = "LEAK" if leak else "SAFE"
    compliance = "NONCOMPLIANT" if not compliant else "COMPLIANT"

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            str(uuid4())[:8],
            datetime.now().isoformat(),
            prompt,
            "[REDACTED]" if leak else response,
            f"{score:.3f}",
            verdict,
            compliance,
            explanation
        ])

    pred_labels.append(1 if leak else 0)

# Accuracy metrics
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)

print("\n🔍 Self-RAG Leak Detection Evaluation")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
