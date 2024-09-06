import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can change to 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit app interface
st.title("AI Article Generator")
st.write("Enter a topic and generate an AI-generated article using GPT-2!")

# User input for the topic
user_topic = st.text_input("Enter the topic for the article", "")

# Generate the article when the user clicks the button
if st.button("Generate Article"):
    if user_topic:
        prompt = f"Write an article on {user_topic}"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate article with GPT-2
        outputs = model.generate(
            inputs,
            max_length=500,  # You can change the length of the generated text
            do_sample=True,  # Enable sampling to get more varied outputs
            top_k=50,        # Limit the sampling pool to top-k tokens
            top_p=0.95,      # Nucleus sampling
            temperature=0.7,  # Controls randomness in sampling
            num_return_sequences=1  # Number of articles to generate
        )

        # Decode the generated article
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the generated text
        st.subheader("Generated Article:")
        st.write(generated_text)
    else:
        st.error("Please enter a topic to generate an article.")
