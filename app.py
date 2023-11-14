import numpy as np
import streamlit as st
from transformers import BertForSequenceClassification,BertTokenizer
import torch


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    model=BertForSequenceClassification.from_pretrained('rajashekarvt/FineTunedBert')
    return tokenizer,model

tokenizer,model=load_model()


user_input=st.text_area('Enter Text To Analyse:')
button=st.button('Analyse')

d={
    1:'Negative',
    0:'Positive'
}

if user_input is not None:
    if st.button:
        test_sample=tokenizer(user_input,padding=True,truncation=True,max_length=512)
        output=model(**test_sample)
        st.write("Logits:",output.logits)
        y_pred=np.argmax(output.logits.detach().numpy(),axis=1)
        st.write("Prediction:",d[y_pred[0]])