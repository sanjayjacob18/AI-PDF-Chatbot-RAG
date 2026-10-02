# 🤖 Production-Grade PDF AI Engine (RAG)

A highly optimized **Retrieval-Augmented Generation (RAG)** backend engineering pipeline built from scratch to parse unstructured documents, generate high-dimensional embeddings, and execute semantic search vectors.

---

## 🛠️ The Tech Architecture Stack
* **Core Runtime Engine:** Python (FastAPI framework)
* **AI Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`) running locally
* **Vector Database Infrastructure:** MongoDB Atlas (Cloud Cluster deployment)
* **Document Processing & Segmentation:** PyPDF Data Stream Parsing & LangChain Recursive Character Splitters

---

## 🏗️ System Workflow Blueprint
1. **Document Ingestion (`POST /upload`):** Reads binary PDF streams, extracts text, and partitions segments into strict 1,000-character semantic blocks with a 200-character boundary overlap.
2. **Embedding Architecture:** Converts text fragments into 384-dimensional mathematical tensor arrays.
3. **Cloud Synchronization:** Pushes structured JSON objects containing chunk identifiers, names, texts, and vector payloads straight into MongoDB Atlas.
4. **Semantic Navigation Search (`GET /query`):** Multi-stage aggregation pipelines match question arrays against document arrays using a cloud-optimized **Cosine Similarity Index** at a search confidence metric threshold above 70%.

---

## 🚀 Local Deployment Setup Guide

### 📦 Package Core Extensions Installation
```bash
pip install fastapi uvicorn pymongo pypdf langchain-text-splitters sentence-transformers python-multipart
```

### 🏃‍♂️ Running the Engine
```bash
python -m uvicorn main:app --reload
```
Once deployed, navigate your local desktop web client to `http://localhost:8000/docs` to run tests via the interactive Swagger interface dashboard.
