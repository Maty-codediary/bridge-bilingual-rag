import os
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import PyMuPDFLoader
MAX_CHARS = 300
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
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s for s in sentences if s.strip()]
        for s in sentences:
            if len(s) > MAX_CHARS:
                safe_sentences.extend(split_long(s))
            else:
                safe_sentences.append(s)
        chunks = make_chunks(safe_sentences)
        for c in chunks:
            all_chunks.append({
                'text': c,
                'source': page.metadata['source'],
                'page': page.metadata['page']+1
            })

print(len(all_chunks))
print(all_chunks[-1])