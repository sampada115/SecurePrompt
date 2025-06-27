from sentence_transformers import SentenceTransformer, util

# Load confidential references
with open("conf_docs.txt", encoding="utf-8") as f:
    conf_docs = [line.strip() for line in f if line.strip()]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
conf_embs = embedder.encode(conf_docs, convert_to_tensor=True)

# Compliance check
banned_keywords = ["undisclosed", "launder", "anonymous"]
def check_compliance(response):
    return not any(k in response.lower() for k in banned_keywords)

def validate_response(prompt, response):
    resp_emb = embedder.encode(response, convert_to_tensor=True)
    sims = util.cos_sim(resp_emb, conf_embs)[0]
    score = sims.max().item()
    match_idx = sims.argmax().item()
    matched_doc = conf_docs[match_idx] if conf_docs else "N/A"
    leak = score >= 0.4
    compliant = check_compliance(response)
    explanation = f"Matched: '{matched_doc}'" if leak else "N/A"
    return leak, compliant, score, explanation
