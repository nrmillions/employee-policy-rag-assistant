# 3–4 minute demo script

Before recording: open Ollama, start Streamlit, open http://localhost:8501, and keep the three sample documents in the folder. Ask one question beforehand to warm the model. Answers may vary slightly; use the actual text on screen. All policies are fictional.

## 0:00–0:30 — Introduce the application

“This is my Employee Policy RAG Assistant. It answers questions using a small collection of employee policy documents. The app uses retrieval-augmented generation, or RAG. It first searches for relevant policy passages, then gives those passages to a language model to produce an answer.”

Show the title, subtitle, question box, and document count.

## 0:30–0:55 — Show the documents

Point to the sidebar, then briefly show the `documents` folder or one TXT file.

“The app has three sample documents covering leave, remote work, and expenses. They are clearly marked fictional and are for a classroom demonstration. I can add text-based PDF files or TXT files to this folder, then click Process / refresh documents.”

## 0:55–1:35 — Straightforward question

Ask: **How many PTO days do full-time employees receive per year?**

“The expected answer is 20 days per calendar year. Under the answer, the app shows the document filename, chunk identifier, and supporting quotation. PDF files would also show the page number. I can check the evidence rather than relying only on the model's wording.”

Show the Sources section and the quote containing 20 days.

## 1:35–2:20 — Another successful question

Ask: **What is the meal reimbursement limit during approved business travel?**

“This answer comes from the expense policy. The meal limit is $50 per day including tax and tips. The supporting excerpt also says an itemized receipt is required and alcohol is not reimbursed. The source makes those conditions visible even when the answer is brief.”

Expand the retrieved-passages section briefly. If the model omits a detail, point out that the source text makes the omission visible.

## 2:20–2:55 — Unsupported question

Ask: **Who is the CEO of Cedar Example Company?**

“The documents do not name a CEO. The assistant says: ‘I could not find this information in the provided documents.’ This is the correct behavior. Even though retrieval returns the closest passages, none provides the requested fact, so the language model should refuse to invent a name. No supporting citation is displayed for that missing fact.”

## 2:55–3:35 — Explain the architecture and evaluation

Show the Mermaid diagram in the README, or keep the retrieved passages open.

“The pipeline starts with text extraction. The app splits documents into overlapping chunks and converts them to local embeddings. It stores the vectors locally, embeds the question, and retrieves the four most similar chunks. The language model sees those retrieved chunks rather than the entire handbook. It answers with sources or declines when there is insufficient evidence.”

“I evaluated 15 questions, including ordinary facts, questions spanning policies, a broad question, and three questions with missing answers. The evaluation report records the real retrieval results and model answers. It also discusses limitations, because citations reduce risk but do not guarantee perfect reasoning.”

## 3:35–3:50 — Finish

“This project demonstrates a complete RAG pipeline in a simple local application. The sample demo runs without an API key using local embeddings and a local language model.”
