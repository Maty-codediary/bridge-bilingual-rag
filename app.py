import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from google.genai import types

load_dotenv()

@st.cache_resource
def load_tools():
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    db = chromadb.PersistentClient(path="chroma_db")
    collection = db.get_collection("reports")
    return gemini, embedder, collection

gemini, embedder, collection = load_tools()
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]   

def ask_gemini(prompt):
    last_error = None
    for model_name in MODELS:                                   
        try:                                                 
            response = gemini.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text, model_name                    
        except Exception as e: 
            print('Failed:', model_name, e)                                 
            last_error = e
    raise last_error
st.title("🌉 Bridge")
st.caption("Ask in English. Get answers from Chinese annual reports (Tencent, BYD, Xiaomi), with the source passage.")

question = st.text_input("Your question", placeholder="What was Tencent's profit in 2025?")

if st.button("Ask") and question:
    with st.spinner("Searching the reports..."):
       q_vector = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_vector, n_results=5)

    # 1. Paste the 5 chunks into one block of text, each with its source + page
    context = ""
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[{meta['source']}, page {meta['page']}]\n{text}\n\n"

    # 2. The instructions for Gemini
    prompt = f"""You are a financial analyst. Answer in English using ONLY the passages below.
    Quote the Chinese sentence you used, translate it, and give its source and page.
    If the answer is not in the passages, say: "I couldn't find this in the documents."

    Passages:
    {context}

    Question: {question}"""

    answer, used_model = ask_gemini(prompt)
    st.markdown(answer)
    st.caption(f"Answered by {used_model}")

    with st.expander("📄 Passages retrieved"):
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            st.markdown(f"**{meta['source']}**, page {meta['page']}")
            st.write(text)

