import os
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("reports")

question = "What was Tencent's profit in 2025?"
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

response = gemini.models.generate_content(model="gemini-3.6-flash", contents=prompt)
print(response.text)
