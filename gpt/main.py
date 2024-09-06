import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

user_topic = input("Enter the topic for the article: ")
prompt = f"Write an article on {user_topic}"

inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    inputs, 
    max_length=500, 
    do_sample=True, 
    top_k=50,     
    top_p=0.95,    
    temperature=0.7, 
    num_return_sequences=1 
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)