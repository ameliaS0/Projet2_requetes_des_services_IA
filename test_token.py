from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
print(pipe("Dis bonjour", max_new_tokens=20)[0]["generated_text"])