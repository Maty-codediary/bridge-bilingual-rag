import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "reports"
TOP_K = 5
SOURCES = {
    "tencent": "data/tencent-annual-report2025.pdf",
    "xiaomi": "data/xiaomi_2025_ar_c.pdf",
    "byd": "data/byd-annual-report2025.pdf",
}

def detect_company(question):
    q = question.lower()
    for name in SOURCES:
        if name in q:              
            return name
    return None

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(COLLECTION)

df = pd.read_csv("eval_set.csv")                              

hits = []
found_pages = []
for _, row in df.iterrows():
    q_vector = model.encode([row["question"]]).tolist()     
    company_q = detect_company(row["question"])
    if company_q:
        results = collection.query(query_embeddings=q_vector, n_results=TOP_K,
                                   where={"source": SOURCES[company_q]})   
    else:
        results = collection.query(query_embeddings=q_vector, n_results=TOP_K)  
    metas = results["metadatas"][0]

 
    expected = [int(p) for p in str(row["pages"]).split(";")]   
    company = row["company"]

    hit = False
    for m in metas:
        if company in m["source"].lower() and m["page"] in expected:   
            hit = True
    hits.append(hit)
    found_pages.append([m["page"] for m in metas])

df["found_pages"] = found_pages
df["hit"] = hits                                         

print(df[["company", "pages", "found_pages", "hit"]])
print("Hit rate:", round(df["hit"].mean() * 100, 1), "%")  
print(df.groupby("company")["hit"].mean() * 100)  
df.to_clipboard_csv("eval_details.csv", index=False)                 
