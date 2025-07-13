from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

def get_chain(vectorstore):
    llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain
