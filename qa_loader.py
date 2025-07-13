
import pandas as pd

def load_qa_pairs(csv_path):
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.capitalize()
    qa_list = []
    for _, row in df.iterrows():
        question = str(row['Question']).strip()
        answer = str(row['Answer']).strip()
        qa_list.append({'question': question, 'answer': answer})
    return qa_list
