import os
import hashlib
from typing import List
import cohere
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Unstructured for advanced document parsing
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

# Load environment variables
load_dotenv()

# Initialize APIs
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ChromaDB (persistent local storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_knowledgebase")


# ================================
# 1. FILE EXTRACTION WITH UNSTRUCTURED.IO
# ================================

def extract_text_with_unstructured(file_path: str) -> List[str]:
    """
    Extract and chunk text using unstructured.io (supports PDF, DOCX, CSV, XLSX, etc.)
    Preserves layout, tables, headers, and structure.
    """
    elements = partition(filename=file_path)
    chunks = chunk_elements(elements, max_characters=1000, overlap=100)
    return [str(chunk) for chunk in chunks]


# For structured files (CSV/XLSX), add row & column-wise parsing
def extract_structured_data(file_path: str) -> List[str]:
    """
    Extract both row-wise and column-wise context from structured files.
    """
    import pandas as pd
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Row-wise: each row as a string
    row_texts = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()

    # Column-wise: unique values per column
    col_texts = [f"{col}: {', '.join(df[col].dropna().astype(str).unique()[:10])}" 
                 for col in df.columns]

    return row_texts + col_texts


# ================================
# 2. DEDUPLICATION & HASHING
# ================================

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_duplicate(text: str) -> bool:
    h = get_hash(text)
    existing = collection.get(where={"hash": h})
    return len(existing['ids']) > 0


# ================================
# 3. INGEST INTO VECTOR DATABASE
# ================================

def ingest_file(file_path: str):
    """
    Ingest any file with deduplication.
    Uses unstructured for general files, custom logic for CSV/XLSX.
    """
    print(f"📄 Processing {file_path}...")

    # Choose extraction method
    if file_path.endswith((".csv", ".xlsx")):
        chunks = extract_structured_data(file_path)
    else:
        chunks = extract_text_with_unstructured(file_path)

    new_texts = []
    new_hashes = []
    new_ids = []

    for i, chunk in enumerate(chunks):
        if not is_duplicate(chunk):
            h = get_hash(chunk)
            new_texts.append(chunk)
            new_hashes.append(h)
            new_ids.append(f"{file_path}_{i}"[:50])  # Chroma ID limit

    if new_texts:
        # Generate embeddings
        embeddings = co.embed(
            texts=new_texts,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=new_texts,
            metadatas=[{"source": file_path, "hash": h} for h in new_hashes],
            ids=new_ids
        )
        print(f"✅ Added {len(new_texts)} new chunks from {file_path}")
    else:
        print(f"⏭️  No new content to add from {file_path}")


# ================================
# 4. TOP-K RETRIEVAL
# ================================

def retrieve_context(query: str, k: int = 3) -> List[str]:
    """
    Retrieve top-k most relevant chunks using Cohere embeddings.
    """
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents"]
    )
    return results["documents"][0]


# ================================
# 5. LLM GENERATION VIA GROQ (LLAMA 3)
# ================================

def generate(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 1024) -> str:
    """
    Generate response using Groq's Llama 3.
    """
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=max_tokens
    )
    return chat_completion.choices[0].message.content


# ================================
# 6. RAG TASKS
# ================================

def rag_query(question: str) -> str:
    context = retrieve_context(question, k=3)
    context_str = "\n\n---\n\n".join(context)
    prompt = f"""
    Use only the following context to answer the question.
    If the answer is not in the context, say 'I don't know'.

    Context:
    {context_str}

    Question: {question}
    Answer:
    """
    return generate(prompt)


def rag_summarize() -> str:
    # Get a sample of stored documents
    all_docs = collection.get()['documents']
    if not all_docs:
        return "No documents in knowledge base."

    sample = " ".join(all_docs[:5])  # Limit context size
    prompt = f"""
    Summarize the key information from the following content:

    {sample}

    Summary:
    """
    return generate(prompt)


def rag_reasoning(question: str) -> str:
    context = retrieve_context(question, k=3)
    context_str = "\n\n---\n\n".join(context)
    prompt = f"""
    Based on the context below, perform logical reasoning to answer the question.

    Context:
    {context_str}

    Question: {question}
    Reasoning and Answer:
    """
    return generate(prompt)


# ================================
# 7. USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    # 🔽 Ingest files (supports: .pdf, .docx, .txt, .csv, .xlsx)
    ingest_file("iv_list.pdf")


    # 🔍 Query
    print("❓ Query Result:")
    print(rag_query("What are the main findings in the report?"))

    # 📊 Summarize
    print("\n📝 Summary:")
    print(rag_summarize())

    # 🤔 Reasoning
    print("\n🧠 Reasoning:")
    print(rag_reasoning("Why did customer satisfaction improve in Q2?"))