
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


def getLLamaresponse(input_text):

    m='llama\llama-2-7b-chat.ggmlv3.q2_K.bin'
    llm=CTransformers(model=m,model_type='llama',config={'max_new_tokens':256,'temperature':0.01})
    
    template="""Write an article for a topic {input_text} ."""
    
    prompt=PromptTemplate(input_variables=["input_text"],template=template)
    
    response=llm(prompt.format(input_text=input_text))
    print(response)
    return response


st.set_page_config(page_title="Article Generator",layout='centered',initial_sidebar_state='collapsed')

st.header("Article Generator")

input_text=st.text_input("Enter the Topic")

submit=st.button("Generate")

if submit:
    st.write(getLLamaresponse(input_text))