# 🌉 Bridge: ask in English, get answers from Chinese annual reports

**Live app:** https://thebridge.streamlit.app/

A large share of what you need to know about Chinese companies is published **only in Chinese**. Analysts who don't read it either miss it or rely on partial English summaries.

**Bridge** lets you ask a question in English and get an answer drawn from the original Chinese documents, **with the exact Chinese passage, its translation, and the page number**, so every answer can be checked.

<!-- Add a screenshot or GIF of the app here -->

---

## What it does

- Indexes **3 Chinese-language annual reports** (Tencent, BYD, Xiaomi, 2025): ~700 pages, **2,958 text chunks**
- Takes a question in **English**, finds the most relevant **Chinese** passages (cross-lingual search)
- Asks **Gemini** to answer **only from those passages**, quote the Chinese sentence, translate it, and cite the page
- Says **"I couldn't find this in the documents"** instead of guessing when the passages don't contain the answer

## How it works

```
PDF (Chinese) → text extraction → sentence chunking → multilingual embeddings → Chroma
↓
English question → embedding → top-5 similar chunks → Gemini (answer + quote + page)
```

| Step | Tool | Why |
|---|---|---|
| Extract | PyMuPDF | pypdf returned garbled Chinese for these PDFs (embedded fonts) |
| Chunk | Custom Python (`re.split` on 。！？) | Chinese has no spaces between words; word-based splitters behave badly |
| Embed | `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) | Maps English and Chinese into the same vector space |
| Store & search | ChromaDB | Local, persistent vector database |
| Answer | Gemini API (free tier), temperature 0 | Grounded answer with citations; temperature 0 for consistent answers |
| App | Streamlit (Community Cloud) | Simple deployment |

## Hard parts & decisions

**1. Garbled Chinese text.** pypdf extracted the Chinese characters as unreadable symbols. I diagnosed it by writing the extracted text to a UTF-8 file (to rule out a terminal display issue), then switched to **PyMuPDF**, which extracted the text correctly.

**2. Chunking Chinese.** I split each page into sentences on Chinese end punctuation (。！？), then grouped sentences into chunks of **≤ 300 characters**. Pages with no punctuation (tables, lists) produced single "sentences" of 900+ characters, so I added a second step that cuts any over-long piece into fixed-size parts.

**3. The model ignored the company name.** The reports say "本集團" ("the Group"), not "Tencent", so a question about Tencent's profit returned BYD's income statement. **Fix: prefix every chunk with its company name in English and Chinese** before embedding.

## Results (experiments so far)

Test question: *"What was Tencent's profit in 2025?"* (target passage: page 5)

| Experiment | Change | Result |
|---|---|---|
| 1 | Prefix each chunk with the company name (EN + 中文) | Right company in top 3: **1/3 → 3/3** |
| 2 | Retrieve 5 chunks instead of 3 | Added unrelated passages; target page still missing → **the embedding model is the bottleneck, not k** |
| 3 | Same passages, repeated runs, temperature 1 → 0 | At temperature 0, **2/3 runs gave the same figures** (page 22); 1/3 answered "I couldn't find this". The retrieved passages only contain a garbled table, so the model hesitates between answering and refusing → **the fix is better retrieval, not prompting** |

A full evaluation (25 questions with known answers) is in progress: see *Next steps*.

## Limitations

- **Tables lose their structure** during extraction: numbers get glued together (e.g. `194,07327,230…`), which makes figures in tables hard to read for both the search and the LLM.
- **Page headers** (company name, page number, section title) stick to the first chunk of each page.
- The light embedding model sometimes misses the most relevant passage (the target sentence on page 5 is not yet in the top 5 for the test question).
- Covers 3 reports only; 2025 data.

## Next steps

- [ ] **Evaluation set:** 25 questions with known answers and pages → retrieval hit rate (is the right page in the top 5?) and answer accuracy
- [ ] **Compare embedding models:** MiniLM vs `bge-m3`
- [ ] **Compare 2 chunking strategies** (e.g. 150 vs 400 characters)
- [ ] **Evaluation tab** in the app showing the results
- [ ] **Model fallback:** retry, then switch to a lighter Gemini model if the main one is overloaded

## Run it locally

```bash
git clone https://github.com/Maty-codediary/bridge-bilingual-rag.git
cd bridge-bilingual-rag
python -m venv .venv
.venv\Scripts\activate # Windows
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key (free at Google AI Studio):

```
GEMINI_API_KEY=your-key-here
```

Then:

```bash
streamlit run app.py
```

The vector database (`chroma_db/`) is included. To rebuild it from the PDFs, put the reports in `data/` and run `python ingest.py`.

## Data

Public 2025 annual reports, Chinese versions, from the companies' investor-relations websites:
- Tencent Holdings (騰訊控股), annual report 2025
- BYD Company (比亚迪), annual report 2025
- Xiaomi Corporation (小米集團), annual report 2025

---

Built by **Maty Traoré**, Big Data, AI & Management student (Audencia × Centrale Nantes), currently on exchange in Shenzhen.

