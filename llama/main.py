
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#model link - https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

def getLLamaresponse(input_text):

    m='llama\llama-2-7b-chat.ggmlv3.q2_K.bin'
    llm=CTransformers(model=m,model_type='llama',config={'max_new_tokens':256,'temperature':0.01})

    template="""Write an article for a topic {input_text}."""
    
    prompt=PromptTemplate(input_variables=["input_text"],template=template)
    
    response=llm(prompt.format(input_text=input_text))
    print(response)
    return response
