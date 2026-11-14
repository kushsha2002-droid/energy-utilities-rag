"""Step 2: Ask questions. Finds the most relevant chunks, sends them to Claude, prints the answer with sources."""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-5-5"
TOP_K = 4  # how many chunks to retrieve per question

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about energy and utilities documents.
Use ONLY the context below. If the answer is not in the context, say you could not find it.

Context:
{context}

Question: {question}

Answer concisely."""
)


def build_chain():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY. Copy .env.example to .env and add your key.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatAnthropic(model=LLM_MODEL)
    return retriever, PROMPT | llm


def _text(content):
    """Claude may return a list of blocks (reasoning + text); keep only the text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")

def answer(retriever, chain, question):
    docs = retriever.invoke(question)  # RETRIEVAL: find relevant chunks
    context = "\n\n".join(d.page_content for d in docs)
    reply = chain.invoke({"context": context, "question": question})  # GENERATION: answer from chunks
    sources = sorted({f"{os.path.basename(d.metadata.get('source', '?'))} (p. {d.metadata.get('page', 0) + 1})" for d in docs})
    return _text(reply.content), sources


def main():
    retriever, chain = build_chain()
    print("Ask a question about your documents (type 'quit' to exit).")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        text, sources = answer(retriever, chain, q)
        print(f"\n{text}\n\nSources: {', '.join(sources)}")


if __name__ == "__main__":
    main()
