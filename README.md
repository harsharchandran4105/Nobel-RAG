# 🧠 Retrieval-Augmented Generation (RAG) System — Nobel Prize 2025 Knowledge Base

> **A Python RAG pipeline that retrieves and generates factual answers using Nobel Prize 2025 data.**

---

## 📘 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system — a hybrid of information retrieval and language generation.  
Instead of relying only on a model’s internal memory, RAG dynamically retrieves relevant text passages from an external dataset before producing an answer.  

The included dataset contains **summarised Nobel Prize 2025 winners and discoveries** in Physics, Chemistry, and Physiology or Medicine.

**Key Highlights**
- 🔍 Local **semantic search** with FAISS and Sentence-Transformers  
- 🧠 **Context-aware generation** using OpenAI GPT (or local Flan-T5 fallback)  
- ⚡ Simple **Streamlit UI** for Q&A  
- 🌐 **FastAPI backend** for programmatic access  
- 📚 Custom dataset: Nobel Prize 2025 laureates and achievements  

---

## 🏗️ Architecture

