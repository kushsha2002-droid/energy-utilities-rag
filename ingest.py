"""Step 1: Read PDFs from ./docs, split into chunks, embed them, and store in a local vector database."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally, no API key needed


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found. Put your PDF files in the 'docs' folder first.")

    pages = []
    for pdf in pdfs:
        loaded = PyPDFLoader(str(pdf)).load()  # one entry per page, source + page number kept
        print(f"Loaded {pdf.name}: {len(loaded)} pages")
        pages.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"Done. Vector database saved in '{DB_DIR}'.")


if __name__ == "__main__":
    main()
