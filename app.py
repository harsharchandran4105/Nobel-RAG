from fastapi import FastAPI
from pydantic import BaseModel
from rag import VectorStore, simple_prompt, generate_answer

app = FastAPI(title="RAG Quickstart API")
store = VectorStore("store")
store.load()

class Query(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag")
def rag_endpoint(q: Query):
    hits = store.search(q.question, k=q.k)
    prompt = simple_prompt(q.question, hits)
    answer = generate_answer(prompt)
    return {"answer": answer, "contexts": hits}