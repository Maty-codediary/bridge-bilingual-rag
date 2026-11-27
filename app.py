import os
import streamlit as st
import chromadb
import pandas as pd
import re
import hashlib
import tempfile
from google import genai
from google.genai import types
from dotenv import load_dotenv
from chunking import split_long, make_chunks
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader



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

@st.cache_resource
def index_pdf(file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)                                   
        path = tmp.name
    pages = PyMuPDFLoader(path).load()[:30]

    texts, metas = [], []
    for page in pages:
        text = page.page_content.replace("\n", "")
        sentences = [s for s in re.split(r'(?<=[。！？])', text) if s.strip()]
        safe = []
        for s in sentences:
            if len(s) > 300:
                safe.extend(split_long(s, 300))
            else:
                safe.append(s)                                   
        for c in make_chunks(safe, 300):
            texts.append(c)
            metas.append({"source": file_name, "page": page.metadata["page"] + 1})

    name = "upload_" + hashlib.md5(file_bytes).hexdigest()[:8]
    col = chromadb.EphemeralClient().get_or_create_collection(name)
    vectors = embedder.encode(texts).tolist()                    
    col.add(ids=[str(i) for i in range(len(texts))], documents=texts,
            embeddings=vectors, metadatas=metas)
    return col

st.title("🌉 Bridge")
st.caption("Ask in English. Get answers from Chinese annual reports (Tencent, BYD, Xiaomi), with the source passage.")

tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])   

with tab_ask:
    question = st.text_input("Your question", placeholder="What was Tencent's profit in 2025?")
    uploaded = st.file_uploader("Or ask about your own Chinese PDF", type="pdf")   
    active = collection
    if uploaded:
        with st.spinner("Reading your PDF (30s)..."):
            active = index_pdf(uploaded.getvalue(), uploaded.name)
            st.caption('Only the first 30 pages are indexed, to kepp it fast on free hosting')
    if st.button("Ask") and question:
        with st.spinner("Searching the document..."):
            q_vector = embedder.encode([question]).tolist()
            results = active.query(query_embeddings=q_vector, n_results=5)

        
            context = ""
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                context += f"[{meta['source']}, page {meta['page']}]\n{text}\n\n"

            
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

with tab_eval:
    st.subheader("How well does Bridge find the right page?")
    st.write("25 questions with known answers. Hit = the correct page is among the 5 passages retrieved.")

    summary = pd.read_csv("eval_summary.csv")
    st.bar_chart(summary.set_index("experiment")["hit_rate"], horizontal=True)   
    st.dataframe(summary, hide_index=True)
    st.caption('The live app uses MiniLM: bge-m3 scored best (72%) but is to large for free hosting. Swapping it in is a one-line change on a bigger server.')

    details = pd.read_csv("eval_details.csv")
    st.dataframe(details[['question', 'hit', 'company', 'pages', 'found_pages']], hide_index=True)
