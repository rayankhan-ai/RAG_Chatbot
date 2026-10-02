# 🤖 AI RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot that allows users to upload one or multiple PDF documents and ask questions about their contents.

The application combines local document embeddings, FAISS vector search, hybrid retrieval, reranking, conversation memory, confidence detection, and a local Ollama LLM to generate grounded answers from the uploaded documents.

---

## 📌 Project Overview

Traditional chatbots generate responses from the knowledge contained in their language model.

This project uses a different approach.

Instead of relying only on the language model, the system first searches the user's documents for relevant information and then provides that information to the language model as context.

This reduces hallucination and allows the chatbot to answer questions based on user-provided documents.

### RAG Workflow

```text
User Question
      ↓
Question Rewriting
      ↓
Hybrid Retrieval
      ↓
FAISS Vector Search
      ↓
Document Filtering
      ↓
Reranking
      ↓
Confidence Detection
      ↓
Relevant Document Chunks
      ↓
Context Construction
      ↓
Ollama LLM
      ↓
Grounded Answer
      ↓
Source References