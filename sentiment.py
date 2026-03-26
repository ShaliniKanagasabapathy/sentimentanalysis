import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis App")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

with st.spinner("Loading model... please wait ⏳"):
    model = load_model()

st.write("Enter your sentence")

user_input = st.text_area("Enter text here:")

if st.button("Analyse"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = model(user_input)[0]

        label = result["label"]
        score = result["score"]

        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

        final_label = label_map.get(label, label)

        st.subheader("Result:")
        st.write(f"Sentiment is {final_label}")
        st.write(f"Confidence Score {score:.2f}")