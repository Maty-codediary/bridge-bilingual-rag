import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader

print('Loading Tencent 2025 annual report...')
loader = PyMuPDFLoader('data/tencent-annual-report2025.pdf')
pages = loader.load()

print(len(pages))
page5= pages[4]
print(page5.page_content[:500])
print(page5.metadata)

with open('page5.txt', 'w', encoding='utf-8')as f:
    f.write(page5.page_content)