from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_fingpt(model_id="tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

def call_fingpt(pipe, prompt, max_new_tokens=100):
    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7)
    response = output[0]["generated_text"]
    return response[len(prompt):].strip() if response.startswith(prompt) else response
