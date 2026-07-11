from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Load model
model_name = "Vamsi/T5_Paraphrase_Paws"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Input text
text = input("Enter text to rephrase: ")

# T5 requires prefix
input_text = "paraphrase: " + text + " </s>"

# 3. Tokenize
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 4. Generate paraphrase
outputs = model.generate(
    input_ids,
    max_new_tokens=60,
    do_sample=True,
    temperature=1.0,
    top_k=120,
    top_p=0.95,
    num_return_sequences=1
)

# 5. Decode output
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nRephrased Text:")
print(paraphrased_text)
