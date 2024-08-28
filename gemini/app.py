
from dotenv import load_dotenv

load_dotenv()  

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])


def get_gemini_response(question):

    response = chat.send_message(question, stream=True)
    return response



st.set_page_config(page_title="GEMINI")

st.header("Article Generartor Using Gemini")

inp = st.text_input("Write your topic: ", key="input")
input="give article on "+ inp

submit = st.button("Generate Article")


if submit:

    response = get_gemini_response(input)
    st.subheader("The Response is")
    for chunk in response:
        print(st.write(chunk.text))
        print("_" * 80)

    st.write(chat.history)