import streamlit as st
from rag import VectorStore, simple_prompt, generate_answer

st.set_page_config(page_title="RAG Quickstart", page_icon="⚡")
st.title("⚡ RAG Quickstart")
st.write("Minimal local RAG with FAISS + SentenceTransformers.")

vs = VectorStore("store")
loaded = vs.load()
if not loaded:
    st.warning("No index found. Run `python ingest.py` first to build the FAISS index from ./data.")
question = st.text_input("Ask a question about your documents:", "What is Retrieval-Augmented Generation?")
k = st.slider("Top-K passages", 1, 10, 5)

if st.button("Run RAG") or "autostart" in st.session_state:
    hits = vs.search(question, k=k) if loaded else []
    st.subheader("Retrieved Context")
    for i, h in enumerate(hits, 1):
        with st.expander(f"[{i}] {h.get('source','unknown')} (score={h.get('score',0):.3f})"):
            st.write(h["text"][:1000])

    prompt = simple_prompt(question, hits)
    with st.spinner("Generating..."):
        answer = generate_answer(prompt)
    st.subheader("Answer")
    st.write(answer)