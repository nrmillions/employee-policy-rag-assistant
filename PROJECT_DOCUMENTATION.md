# Employee Policy RAG Assistant — Class Project Documentation

## Project Overview

This project is a small web application that answers questions about employee policy documents. It uses genuine Retrieval-Augmented Generation (RAG): retrieve a few relevant passages first, then give only those passages and the question to a language model. Answers include supporting quotes and filenames, with PDF page numbers when available.

## Problem / Use Case

An employee may need to know how much PTO is available or how to claim an expense. Searching several handbook documents manually takes time. This assistant helps locate and explain policy information while making the original evidence visible.

## Target User

The target user is an employee, HR staff member, or student demonstrating document-based question answering. The interface has a question box, an Ask button, a document count, an answer, and sources. No login is required.

## Knowledge Corpus

The initial corpus contains three fictional Cedar Example Company policies: leave, remote work, and expenses. They cover PTO, carryover, parental leave, remote eligibility, working hours, security, home office purchases, and reimbursements. Every document is clearly labeled SAMPLE / FICTIONAL. Additional PDF or TXT files can be placed in `documents/`.

## Ingestion and Cleaning

The app reads UTF-8 TXT files directly and uses pypdf to extract each PDF page. It normalizes whitespace, preserves filenames and one-based PDF page numbers, and warns about unreadable files or pages without text. It does not perform OCR. A hash of file names and contents detects changes, so the local index can be reused when documents are unchanged.

## Chunking Strategy

Text is split into passages of up to 900 characters with approximately 150 characters of overlap. Where possible, a chunk ends at a sentence boundary or complete word. Chunks never span PDF pages. Each chunk has an ID. Overlap helps retain context at passage boundaries, but can also produce repeated information in search results.

## Embedding Model

FastEmbed runs the `BAAI/bge-small-en-v1.5` model locally. It converts each passage and each question into a 384-number vector representing meaning. Document passages and search questions use the model's passage/query methods. This is semantic vector retrieval, not a keyword matching shortcut. The model is downloaded once and cached.

## Vector Store

Normalized embeddings are stored as a NumPy matrix in `.cache/vectors.npy`. The corresponding text, filenames, page numbers, IDs, and corpus signature are stored in `.cache/index.json`. An exact search over this local matrix is appropriate for a tiny corpus. FAISS or Chroma would be useful for larger projects, but an additional database library is unnecessary here.

## Retrieval Strategy

The question is embedded and normalized. A matrix multiplication computes cosine similarity between its vector and every stored passage vector. The four highest-scoring chunks become the context. Scores measure similarity, not whether an answer is true. There is no fixed cutoff because an arbitrary similarity threshold cannot reliably detect all unsupported questions.

## LLM / Generation

The default LLM is `qwen2.5:3b` through the already-installed local Ollama application. It runs with temperature zero and structured JSON output. Only the retrieved chunks are supplied. This allows the complete installed demo to run without an API key. OpenAI `gpt-4o-mini` is available as an optional API alternative using `OPENAI_API_KEY` in `.env`. The OpenAI path uses the Responses API and disables response storage with `store=False`; it was implemented but cannot be tested without a user-supplied key.

## Hallucination Prevention

The system instruction says to answer using only the supplied context and to refuse unsupported questions. The exact refusal is: “I could not find this information in the provided documents.” The model must return supporting chunk IDs and exact quotations. Code checks that quotations exist in the retrieved chunks and that cited IDs are valid before displaying a supported answer. Instructions inside documents are treated as untrusted data. These measures reduce hallucination but do not prove that every answer follows logically from its quote; a model can still misinterpret a policy.

## Citation Strategy

Answers use bracketed chunk IDs, such as `[3]`. The Sources section maps those IDs to a filename, PDF page when available, chunk ID, and exact supporting excerpt. Only sources selected by the model appear as supporting citations. A separate expandable section shows all retrieved passages. A refusal has no supporting source; retrieved passages remain inspectable and are not presented as proof of the missing fact.

## Evaluation Method

Fifteen questions test ordinary facts, information across policies, a broad question, and three missing facts. `evaluate.py` uses the same retrieval and generation functions as the UI. It records actual answers and retrieved chunks, then creates a report for human review. A reviewer compares each factual claim against the retrieved text, marks groundedness Yes/No, and scores retrieval quality from 1 to 5. Unsupported questions test whether generation abstains even when similarity retrieval still returns passages. The evaluation report explains its scoring rubric and contains failure analysis. This small sample is a classroom check, not a general accuracy benchmark.

## Iterations / Design Decisions

The design prioritizes a working demo and easy explanation. Streamlit removes the need to write a separate frontend. Local embeddings avoid an embedding API key. NumPy avoids a database account. Ollama was already installed, making local generation practical; a small model was downloaded to complete the end-to-end pipeline. Exact-quote verification catches unreliable citations. The extra evaluation script exists because reproducible evidence is useful for the class assignment. No authentication, agents, Docker, or external database was added.

## Limitations

- Scanned PDFs need OCR before ingestion; complex PDF layouts may extract poorly.
- English embeddings and fictional English policies were tested; other languages were not evaluated.
- The small local LLM may omit details, refuse valid questions, or cite evidence that does not support its reasoning.
- Four passages may miss relevant clauses in larger corpora. Overlap can fill results with similar passages.
- There is no document precedence rule for contradictory or outdated policies.
- The app does not know an employee's role, service history, or personal circumstances.
- Exact quotation validation verifies evidence location, not logical entailment.
- The vector store is designed for small collections and a local class demo.
- Initial installation/model downloads need internet, and local generation requires Ollama to be running.
- OpenAI generation requires a key and API credits; that optional path was not live-tested here.

## What I Learned

The points below are presentation prompts; adapt them to describe your own experience:

- Retrieval and generation are different steps. A fluent answer is not enough without evidence.
- Chunk size affects what context can be found and understood.
- Embeddings help retrieve related wording even when a question uses different words.
- Showing sources makes an answer easier to audit.
- Unsupported questions are important tests: the assistant should sometimes decline to answer.
- A compact system can demonstrate the complete RAG architecture without complex infrastructure.
