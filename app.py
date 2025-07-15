import streamlit as st
from qa_loader import load_qa_pairs
from vector_store import create_or_load_vectorstore
from chatbot_chain import get_chain

st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title(" Medical Chatbot")
st.markdown("Ask health-related questions based on trained data.")

# Load and prepare
qa_data = load_qa_pairs("medical_chatbot.csv")
vectorstore = create_or_load_vectorstore(qa_data)
qa_chain = get_chain(vectorstore)

# Chat interface
query = st.text_input("Ask your question here:")

if query:
    with st.spinner("Searching..."):
        response = qa_chain.run(query)
        st.success(response)
