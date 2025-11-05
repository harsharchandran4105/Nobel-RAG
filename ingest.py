import os
from pypdf import PdfReader
from rag import VectorStore, chunk_text

DATA_DIR = "data"
STORE_DIR = "store"

def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    docs.append((path, fp.read()))
            elif f.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    docs.append((path, text))
                except Exception as e:
                    print(f"[warn] Could not parse PDF: {path} ({e})")
    return docs

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    docs = load_documents()
    if not docs:
        # Seed sample if empty
        sample_path = os.path.join(DATA_DIR, "sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Retrieval-Augmented Generation (RAG) augments large language models with external knowledge by retrieving relevant context and feeding it into the prompt before generation.")
        docs = [(sample_path, open(sample_path, "r", encoding="utf-8").read())]

    chunks = []
    for path, text in docs:
        if not text.strip():
            continue
        pieces = chunk_text(text, source=path, chunk_size=180, overlap=30)
        chunks.extend(pieces)

    vs = VectorStore(STORE_DIR)
    vs.build(chunks)
    print(f"[ok] Indexed {len(chunks)} chunks from {len(docs)} documents into {STORE_DIR}/")

if __name__ == "__main__":
    main()