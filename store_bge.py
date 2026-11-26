import json, numpy as np, chromadb
data = json.load(open("chunks.json", encoding="utf-8"))
vecs = np.load("bge_vectors.npy")
col = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("reports_bge")
col.upsert(ids=data["ids"], documents=data["texts"], embeddings=vecs.tolist(), metadatas=data["metadatas"])
print("Stored:", col.count())