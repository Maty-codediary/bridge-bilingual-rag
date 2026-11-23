import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client= chromadb.PersistentClient(path='chroma_db')
collection = client.get_or_create_collection('reports')

question = "What was Tencent's profit in 2025?"
q_vector = model.encode([question]).tolist()

results = collection.query(query_embeddings=q_vector, n_results=3)

for text, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(meta['source'], 'page', meta['page'])
    print(text[:150])
    print('---')