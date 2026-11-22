from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

sentences = [
    "What was Tencent's profit in 2025?",
    "騰訊二零二五年的盈利為人民幣2,248億元。",
    "BYD sold many electric cars.",
]
vectors = model.encode(sentences)

print(vectors.shape)
print(util.cos_sim(vectors[0], vectors[1]))   
print(util.cos_sim(vectors[0], vectors[2]))   