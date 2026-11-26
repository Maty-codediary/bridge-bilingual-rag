import os
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
MAX_CHARS = 300
COMPANIES = {   "data/tencent-annual-report2025.pdf": "Tencent 騰訊 2025 annual report",
    "data/xiaomi_2025_ar_c.pdf": "Xiaomi 小米 2025 annual report",
    "data/byd-annual-report2025.pdf": "BYD 比亚迪 2025 annual report",

}

XIAOMI_PUNCT = {"灱": "，", "˿": "。", "炀": "；", "̈": "「", "̉": "」",
                "灿": "：", "灭": "（", "灮": "）"}

def fix_xiaomi(text):
    out = []
    for ch in text:
        code = ord(ch)                          
        if ch in XIAOMI_PUNCT:
            out.append(XIAOMI_PUNCT[ch])       
        elif 0x1A00 <= code <= 0x6F00:
            out.append(chr(code + 0x303E))      
        else:
            out.append(ch)
    return "".join(out)                      


def make_chunks(sentences, max_char = 300):
    chunks = []
    current = ""
    for s in sentences:
        if len(current)+ len(s)> max_char and current:
         chunks.append(current)
         current = ''
        current += s
    if current:
       chunks.append(current)
    return chunks

def split_long(s, max_char=300):
    pieces=[]
    for i in range(0, len(s), max_char):
        pieces.append(s[i:max_char +i])
    return pieces

print('Loading Tencent, Xiaomi, BYD 2025 annual report...')
files = ['data/tencent-annual-report2025.pdf', 'data/xiaomi_2025_ar_c.pdf', 'data/byd-annual-report2025.pdf']
all_chunks = []
for f in files:
    loader = PyMuPDFLoader(f)
    pages = loader.load() 
    for page in pages:
        safe_sentences = []
        text = page.page_content
        text = text.replace('\n', '')
        if "xiaomi" in f.lower():
            text = fix_xiaomi(text)
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s for s in sentences if s.strip()]
        for s in sentences:
            if len(s) > MAX_CHARS:
                safe_sentences.extend(split_long(s, MAX_CHARS))
            else:
                safe_sentences.append(s)
        chunks = make_chunks(safe_sentences, MAX_CHARS)
        for c in chunks:
            all_chunks.append({
                'text': c,
                'source': page.metadata['source'],
                'page': page.metadata['page']+1
            })

print(len(all_chunks))
print(all_chunks[-1])

import chromadb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client= chromadb.PersistentClient(path='chroma_db')
collection = client.get_or_create_collection('reports')

texts = [COMPANIES[c['source']] + ':' + c['text'] for c in all_chunks]
metadatas = [{'source': c['source'], 'page': c['page']} for c in all_chunks]
ids = [str(i) for i in range(len(all_chunks))]

vectors = model.encode(texts, show_progress_bar= True)

BATCH = 5000
for i in range(0, len(ids), BATCH):
    collection.upsert(ids=ids[i:i+BATCH], documents=texts[i:i+BATCH],
                      embeddings=vectors[i:i+BATCH].tolist(), metadatas=metadatas[i:i+BATCH])
print('Stored:', collection.count())