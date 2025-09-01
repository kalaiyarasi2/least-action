# RAG Pipeline: Multiformat Document QA, Summarization & Reasoning

## Project Purpose & Motivation

This project provides a robust Retrieval-Augmented Generation (RAG) pipeline capable of extracting, deduplicating, embedding, and querying information from multiple file formats. It enables grounded Q&A, advanced summarization, and reasoning over enterprise or research documents, removing the pain of scattered, unsearchable knowledge bases.[3][1]

***

## System Architecture

```
File Input (PDF, DOCX, TXT, CSV, XLSX)
      │
      ▼
Extraction & Chunking (Unstructured.io/Pandas)
      │
      ▼
Deduplication (MD5)
      │
      ▼
Embedding (Cohere)
      │
      ▼
Vector DB (ChromaDB)
      │
      ▼
Top-k Retrieval
      │
      ▼
Prompt Construction
      │
      ▼
LLM Generation (Groq Llama 3)
      │
      ▼
Answer / Summary / Reasoning
```
- Modular design with clear separation of extraction, deduplication, embedding, and LLM query.[4][1]

***

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone git@github.com:kalaiyarasi2/least-action.git
   cd least-action
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Requirements include: `cohere`, `chromadb`, `groq`, `unstructured`, `pandas`, `dotenv`

3. **Create `.env` File**
   ```env
   COHERE_API_KEY=your-cohere-api-key
   GROQ_API_KEY=your-groq-api-key
   ```
   (Place this file in your project root.)

4. **Directory and Files**
   - All processed data is stored in `./chroma_db`
   - Place your input files in the working directory.

***

## Usage Example

```python
python main.py
```
- Edit the filename in the `__main__` section for your test file:
  ```python
  ingest_file("your_file.pdf")
  print(rag_query("What are the main findings in the report?"))
  print(rag_summarize())
  print(rag_reasoning("Why did customer satisfaction improve in Q2?"))
  ```

***

## Component Descriptions

| Function                        | Purpose                                                                 |
|----------------------------------|-------------------------------------------------------------------------|
| `extract_text_with_unstructured` | Extracts/chunks text from unstructured files (e.g., PDF, DOCX, TXT)     |
| `extract_structured_data`        | Row- and column-wise extraction for CSV/XLSX                            |
| `get_hash` / `is_duplicate`      | Runs deduplication on text chunks using MD5                             |
| `ingest_file`                    | Complete ingestion, chunking, embedding, and upserting to ChromaDB      |
| `retrieve_context`               | Returns top-k relevant context using semantic search                     |
| `generate`                       | Sends prompt to Llama 3 (Groq) and fetches response                     |
| `rag_query` / `rag_summarize` / `rag_reasoning` | Entry points for user to ask, summarize, or reason                    |

***

## Supported Input Formats

- PDF, DOCX, TXT, CSV, XLSX
- **Limitation:** Extremely large files may require increased system memory or custom chunking.

***

## Deduplication

- All text chunks are hashed using MD5.
- Only unique content is embedded and added, reducing storage costs and increasing query precision.

***

## Embedding & Vector Store

- Uses Cohere `embed-english-v3.0` model for embeddings.
- Stores vectors in ChromaDB locally for high-speed, persistent access.[5][1]
- Metadata includes source filename and hash for traceability.

***

## Main RAG Tasks

- **Query:** Ask questions and get answers rooted in your documents.
- **Summarize:** Get automatic summaries of stored knowledge.
- **Reason:** Multi-step logic and synthesis tasks.

***

## LLM Integration

- Powered by Groq's hosted Llama 3 models (adjustable in `generate`).
- Prompting strategy keeps context and instructions clear for robust results.

***

## Extensibility

- Add more input formats by updating extraction utilities.
- Swap out LLM (e.g., to Gemini) by replacing Groq API usage in `generate`.
- Add support for multimodal data with Unstructured.io extensions.

***

## Testing

- Start with a small example file and test all three core RAG methods.
- Inspect ChromaDB for content presence, check answers for fidelity.
- Edge cases: test duplicate files, files with unusual encoding, or empty files.

***

## References & Further Reading

- [LangChain RAG Overview][6]
- [ChromaDB Documentation][7]
- [Cohere Embeddings][5]
- [Best Practices: RAG App Design][3]
