# 🌉 Bridge: ask in English, get answers from Chinese annual reports

**Live app:** https://thebridge.streamlit.app/

A large share of what you need to know about Chinese companies is published **only in Chinese**. Analysts who don't read it either miss it or rely on partial English summaries.

**Bridge** lets you ask a question in English and get an answer drawn from the original Chinese documents, **with the exact Chinese passage, its translation, and the page number**, so every answer can be checked. You can also upload your own Chinese PDF.

<!-- Add a screenshot or GIF of the app here -->

---

## What it does

- Indexes **3 Chinese-language annual reports** (Tencent, BYD, Xiaomi, 2025): ~950 pages, **3,049 text chunks**
- Takes a question in **English** and finds the most relevant **Chinese** passages (cross-lingual search)
- Asks **Gemini** to answer **only from those passages**, quote the Chinese sentence, translate it and cite the page
- Says **"I couldn't find this in the documents"** instead of guessing when the passages don't contain the answer
- **Upload your own Chinese PDF** and ask questions about it
- An **Evaluation tab** shows how well retrieval works, measured on a test set

## How it works

```
PDF (Chinese) → text extraction → sentence chunking → multilingual embeddings → Chroma
                                                                                  ↓
English question → embedding → top-5 similar chunks → Gemini (answer + quote + page)
```

| Step | Tool | Why |
|---|---|---|
| Extract | PyMuPDF (+ a custom decoder for one report) | pypdf returned garbled Chinese; one report needed a font-encoding fix (see below) |
| Chunk | Custom Python (`re.split` on 。！？), ≤ 300 characters | Chinese has no spaces between words; word-based splitters behave badly |
| Embed | `paraphrase-multilingual-MiniLM-L12-v2` (live app) · `BAAI/bge-m3` (best in evaluation) | Map English and Chinese into the same vector space |
| Store & search | ChromaDB | Local, persistent vector database |
| Answer | Gemini API, temperature 0, with automatic fallback (3.8 Flash → 3.6 Flash → 3.5 Flash-Lite) | Grounded answers with citations; stays up when a model is overloaded |
| App | Streamlit (Community Cloud) | Simple deployment |

## Evaluation

I built a test set of **25 questions with known answers and pages** (10 Tencent, 9 BYD, 6 Xiaomi), including 2 trap questions (Q4 vs full year; "did BYD's profit grow?" when it actually fell). Metric: **hit rate** = the expected page is among the 5 passages retrieved.

| # | Setup | Hit rate | Tencent | BYD | Xiaomi |
|---|---|---|---|---|---|
| 1 | Baseline (MiniLM, 300 chars) | 40% | 40% | 56% | 17% |
| 2 | **Fix Xiaomi font encoding** | **56%** | 40% | 44% | **100%** |
| 3 | Smaller chunks (150 chars) | 56% | 50% | 44% | 83% |
| 4 | Bigger chunks (400 chars) | 40% | 40% | 44% | 33% |
| 5 | **Stronger embedding model (bge-m3)** | **72%** | 50% | 78% | 100% |

Also tested: filtering the search by company (Chroma `where`) → **no change**, which showed the misses were wrong pages inside the right report, not confusion between companies.

## Hard parts & decisions

**1. A whole company was invisible to the search.** The evaluation showed Xiaomi at 17%, with the same pages returned for every question. The cause: the Xiaomi PDF's font has no Unicode mapping, so both pypdf and PyMuPDF returned gibberish. I found that every Chinese character was shifted by a fixed offset (+0x303E) and wrote a small decoder (plus a lookup table for punctuation). Xiaomi went from **17% to 100%**, overall from **40% to 56%**. I would never have found this without measuring.

**2. Chunking Chinese.** I split each page into sentences on Chinese end punctuation (。！？) and grouped them into chunks of ≤ 300 characters. Pages with no punctuation (tables, lists) produced 900+ character "sentences", so a second step cuts any over-long piece. 150 characters helped short factual questions but lost context elsewhere; 400 diluted the match. 300 was the best balance.

**3. The model ignored the company name.** The reports say "本集團" ("the Group"), not "Tencent", so a question about Tencent's profit returned BYD's income statement. Prefixing every chunk with its company name in English and Chinese fixed it (right company in the top 3: 1/3 → 3/3).

**4. Best model vs deployable model.** bge-m3 scored best (72% vs 56%), but it is ~2.2 GB: too large for free hosting, and encoding 3,000 chunks would have taken ~4.5 hours on my laptop's CPU. I encoded it on a free Colab GPU in minutes to evaluate it. The live app keeps MiniLM; switching models is a one-line change on a bigger server.

**5. Staying up when the AI provider is down.** During testing, Gemini returned "503 overloaded" errors. The app now tries 3 models in order, so the user still gets an answer.

## Limitations

- **Tables lose their structure** during extraction: numbers get glued together, which makes figures in tables hard to read for both search and LLM.
- **Wrong page inside the right report** is now the main failure: the right company is found, but not always the exact page (e.g. Tencent's annual profit sentence on page 5).
- A company filter fails when the question doesn't name the company (e.g. "WeChat", "Fangchengbao").
- The fallback only covers one provider: if all Gemini models are overloaded at the same time, the app shows an error.
- Covers 3 reports, 2025 data; uploaded PDFs with very unusual fonts may need a custom decoder like Xiaomi's.

## Run it locally

```bash
git clone https://github.com/Maty-codediary/bridge-bilingual-rag.git
cd bridge-bilingual-rag
python -m venv .venv
.venv\Scripts\activate          # Windows
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

The vector database (`chroma_db/`) is included. To rebuild it, put the reports in `data/` and run `python ingest.py`. To re-run the evaluation: `python evaluate.py`.

## Data

Public 2025 annual reports, Chinese versions, from the companies' investor-relations websites:
- Tencent Holdings (騰訊控股), annual report 2025
- BYD Company (比亚迪), annual report 2025
- Xiaomi Corporation (小米集團), annual report 2025



