
import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_or_load_vectorstore(qa_list, save_path="faiss_index"):
    if os.path.exists(os.path.join(save_path, "faiss_store.pkl")):
        print(" Loading saved FAISS index...")
        with open(os.path.join(save_path, "faiss_store.pkl"), "rb") as f:
            return pickle.load(f)

    print(" Creating FAISS index from scratch...")
    texts = [item['question'] for item in qa_list]
    metadatas = [{"answer": item["answer"]} for item in qa_list]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "faiss_store.pkl"), "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore
