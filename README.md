# Energy & Utilities Document Q&A (RAG)

A Retrieval-Augmented Generation (RAG) app that answers plain-English questions about US energy and utilities reports and cites the source file and page for every answer. If the answer isn't in the documents, it says so instead of guessing.

**Stack:** Python, LangChain, Chroma (vector database), sentence-transformers (all-MiniLM-L6-v2 embeddings), Claude API

## How it works
1. **Ingest:** PDFs are loaded page by page, split into ~1,000-character overlapping chunks, embedded locally, and stored in a Chroma vector database.
2. **Retrieve:** The question is embedded and the 4 most similar chunks are retrieved.
3. **Generate:** The chunks are passed to Claude through a LangChain prompt that restricts the answer to the retrieved context. The answer is returned with source file names and page numbers.

## Documents used
Public reports from US energy regulators and agencies (not included in this repo):
- NERC 2024 Long-Term Reliability Assessment
- FERC State of the Markets (2022 and 2024)
- EIA Annual Energy Outlook 2025 (narrative)

## Run it
**Google Colab (easiest):** open `energy_rag_colab.ipynb` in Colab, add your `ANTHROPIC_API_KEY` as a Colab secret, upload your PDFs, and run the cells top to bottom.

**Locally:**
```bash
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
# put PDFs in docs/
python ingest.py
python ask.py
```

## Example outputs
**Q: What are the main risks or challenges discussed in these documents?**
Grouped answer covering resource adequacy and supply (slower resource additions, shortfall risk in extreme conditions), energy adequacy (aging resources, contract-end exits, delayed new resources, transmission lead times), fuel risks, and demand-side factors (data centers, EV supply chain loads, climate-sensitive load).
*Sources: NERC LTRA 2024 (pp. 12, 81, 145), FERC State of the Markets 2024 (p. 11)*

**Q: How is data center growth expected to affect electricity demand?**
Demand growth is higher than at any point in the past two decades, and large loads such as data centers and crypto/AI facilities are connecting quickly, making forecasting and planning harder. Cites PJM data center growth rates of over 300% in some cases (Loudoun County, Virginia). EVs and heat pumps are also named as demand drivers.
*Sources: NERC LTRA 2024 (pp. 8, 38, 93)*

**Q: What did FERC say about natural gas price volatility?**
The app reports it could not find a FERC statement about volatility itself, then returns related facts it did find: Henry Hub averaged $2.53/MMBtu in 2023, down 60% from $6.38 in 2022, and FERC issued a revised price index policy statement on April 21, 2022.
*Sources: FERC State of the Markets 2022 (p. 12), 2024 (pp. 25, 33)*

## Limitations and next steps
- Answers depend on retrieval quality; the top-4 chunk setting is a simple default and was not tuned.
- No formal evaluation yet; next step is a labeled set of test questions to measure retrieval accuracy.
- Planned: expose as a REST API (FastAPI) with API-key authentication, containerize with Docker, and deploy to AWS or Azure.
