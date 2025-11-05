# Retrieval-Augmented Generation (RAG) — Nobel Prize 2025

A compact Retrieval-Augmented Generation (RAG) system that answers factual questions using a curated **Nobel Prize 2025** dataset.  
It performs dense retrieval with **FAISS** + **Sentence-Transformers** and generates answers with **OpenAI GPT** (or a local fallback).

---

## Overview

Traditional LLMs rely on static pretraining and can miss fresh or domain-specific facts.  
**RAG** improves factuality by retrieving relevant text from an external corpus and feeding it to the generator.

This repository demonstrates a minimal, production-style RAG pipeline backed by a 2025 Nobel dataset.

---

## Features

- **Semantic retrieval:** FAISS + `all-MiniLM-L6-v2` embeddings  
- **Generation backends:** OpenAI GPT (if key provided) or local Flan-T5 fallback  
- **Interfaces:** Streamlit UI, FastAPI REST API, and a simple CLI  
- **Custom dataset:** Nobel Prize 2025 summaries formatted for precise Q&A  
- **Local first:** All retrieval runs on your machine (no cloud vector DB required)

