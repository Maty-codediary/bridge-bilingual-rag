import os
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import PyMuPDFLoader
MAX_CHARS = 300
print('Loading Tencent 2025 annual report...')
loader = PyMuPDFLoader('data/tencent-annual-report2025.pdf')
pages = loader.load()

print(len(pages))
page5= pages[4]

text = page5.page_content
text = text.replace('\n', '')
sentences = re.split(r'(?<=[。！？])', text)
sentences = [s for s in sentences if s.strip()]
print(len(sentences))
for i,s in enumerate(sentences[:5]):
    print(i, s)

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

chunks = make_chunks(sentences)
print(len(chunks))
for i, c in enumerate(chunks):
   print(i,len(c), c)

def split_long(s, max_char=300):
    pieces=[]
    for i in range(0, len(s), max_char):
        pieces.append(s[i:max_char +i])
    return pieces


for page in pages[:3]:
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
    print(page.metadata['page'], len(chunks))



