import os
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class VectorStore:
    def __init__(self, path: str = "store"):
        self.path = path
        self.index_path = os.path.join(path, "faiss.index")
        self.meta_path = os.path.join(path, "meta.pkl")
        os.makedirs(self.path, exist_ok=True)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        self.index = None
        self.metadata = []

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True), dtype="float32")

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def build(self, chunks: List[Dict[str, Any]]):
        embeddings = self.embed([c["text"] for c in chunks])
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        self.metadata = chunks
        self.save()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load()
        q = self.embed([query])
        scores, idx = self.index.search(q, k)
        results = []
        for i, score in zip(idx[0], scores[0]):
            if i == -1:
                continue
            item = dict(self.metadata[i])
            item["score"] = float(score)
            results.append(item)
        return results


def simple_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    header = "You are a helpful assistant. Use ONLY the provided context to answer succinctly.\n\n"
    ctx = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    q = f"\n\nQuestion: {question}\nAnswer:"
    return header + "Context:\n" + ctx + q


def generate_answer(prompt: str) -> str:
    # Try OpenAI if available, else fallback to local transformers
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        # fall back below
        pass

    # Fallback to small local model
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        return "Generation backend not available. Please set OPENAI_API_KEY or install transformers."


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append({"text": chunk, "source": source})
        i += chunk_size - overlap
    return chunks