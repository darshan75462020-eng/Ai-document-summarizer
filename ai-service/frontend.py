import streamlit as st
import requests

st.title("AI Document Summarizer")

st.write("Upload a PDF and get an AI-generated summary.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Summarize Document"):
        with st.spinner("Generating summary..."):

            files = {"file": uploaded_file.getvalue()}

            response = requests.post(
                "http://127.0.0.1:8000/summarize",
                files=files
            )

            if response.status_code == 200:

                data = response.json()

                summary = data["summary"]
                bullets = data["bullets"]
                keywords = data["keywords"]

                st.subheader("Summary")
                st.write(summary)

                st.subheader("Key Points")
                st.markdown(bullets)

                st.subheader("Keywords")
                st.write(keywords)

            else:
                st.error("Something went wrong")