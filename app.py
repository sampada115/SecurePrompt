from validator.self_rag import is_response_leaking
from validator.compliance import check_keywords
from datetime import datetime
import csv

def call_llm(prompt):
    if "benchmark" in prompt.lower():
        return "Benchmark manipulation during crises yielded unfair profit."
    return "This depends on market conditions and regulatory norms."

def validate_response(prompt, response):
    leak, score, matched = is_response_leaking(response)
    compliance_issue = check_keywords(response)

    if leak or compliance_issue:
        with open("blocked_responses.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), prompt, response, score, matched])

        reason = "Leak Detected" if leak else "Compliance Violation"
        return {
            "status": "blocked",
            "reason": reason,
            "score": score,
            "matched_doc": matched,
            "cleaned_response": "[REDACTED]"
        }

    return {
        "status": "delivered",
        "response": response,
        "score": score
    }

if __name__ == "__main__":
    prompt = "How can benchmark manipulation be used during crises?"
    response = call_llm(prompt)
    result = validate_response(prompt, response)

    print(f"Prompt: {prompt}")
    print(f"Response: {result['response']}")
    print(f"Status: {result['status']}")
    if result['status'] == "blocked":
        print(f"Matched Doc: {result['matched_doc']}")
        print(f"Similarity Score: {result['score']}")
