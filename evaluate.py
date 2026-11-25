import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "reports"
TOP_K = 5

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(COLLECTION)

df = pd.read_csv("eval_set.csv")                              

hits = []
found_pages = []
for _, row in df.iterrows():
    q_vector = model.encode([row["question"]]).tolist()     
    results = collection.query(query_embeddings=q_vector, n_results= TOP_K)   
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
