# 📚 RAG (Retrieval-Augmented Generation) Project

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions from their documents.

Instead of directly asking an LLM to answer a question, the system:

1. Loads documents/PDFs
2. Extracts text from the documents
3. Splits the text into smaller chunks
4. Converts chunks into vector embeddings
5. Stores embeddings in **Pinecone Vector Database**
6. Searches for the most relevant chunks for a user's question
7. Sends the retrieved context to an LLM
8. Generates a relevant answer using **Groq LLM**

### 🔄 RAG Pipeline

```text
             Document / PDF
                    ↓
              Text Extraction
                    ↓
               Text Splitting
                    ↓
                Text Chunks
                    ↓
              Embeddings Model
                    ↓
             Vector Embeddings
                    ↓
             Pinecone Database
                    ↓
               User Question
                    ↓
          Question → Embedding
                    ↓
          Similarity Search
                    ↓
          Relevant Chunks/Context
                    ↓
                Groq LLM
                    ↓
              Final Answer
```

---

## 🎯 Objective

The main objective of this project is to build a question-answering system that can retrieve relevant information from custom documents and generate answers based on that information.

This approach helps reduce the problem of an LLM generating answers without having access to the user's private/custom documents.

---

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **PyPDF**
* **Sentence Transformers**
* **Pinecone**
* **Groq API**
* **Vector Embeddings**
* **RAG Architecture**
* **dotenv**
* **Jupyter Notebook / VS Code**

---

## 📂 Project Structure

```text
RAG_PROJECT/
│
├── data/
│   └── document.pdf
│
├── myenv/
│
├── .env
├── .gitignore
├── requirements.txt
├── rag.ipynb
└── README.md
```

> The exact file structure may change as the project is further modularized.

---

# 🔹 Step 1: Load Environment Variables

API keys are stored in a `.env` file instead of directly writing them inside the Python code.

Example:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
```

### Why?

API keys are sensitive credentials and should not be exposed in source code or GitHub.

---

# 🔹 Step 2: Load PDF Documents

The PDF document is loaded and its text is extracted.

Example:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/document.pdf")

documents = loader.load()
```

The loader converts the PDF into document objects containing:

* Text
* Metadata
* Page information

---

# 🔹 Step 3: Split Documents into Chunks

Large documents are divided into smaller pieces called **chunks**.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
```

### Why chunking?

LLMs and embedding models cannot efficiently process very large documents as one piece.

Chunking allows the system to:

* Search smaller sections
* Improve retrieval
* Reduce unnecessary context
* Improve answer relevance

### Important Parameters

**chunk_size**

Maximum approximate size of each chunk.

**chunk_overlap**

Number of characters shared between consecutive chunks.

Example:

```text
Chunk 1: A B C D E
Chunk 2:         D E F G H
                ↑
             overlap
```

---

# 🔹 Step 4: Generate Embeddings

Each text chunk is converted into a numerical vector using an embedding model.

Example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)
```

An embedding represents the **semantic meaning** of text in numerical form.

For example:

```text
"Python is a programming language"
                ↓
       [0.12, -0.34, 0.56, ...]
```

Texts with similar meanings generally have similar vectors.

---

# 🔹 Step 5: Store Vectors in Pinecone

Pinecone is used as the vector database.

```python
from pinecone import Pinecone

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("sentence-transformer-index")
```

The generated vectors are uploaded to the Pinecone index along with metadata.

Example vector structure:

```python
(
    "0",
    embeddings[0].tolist(),
    {
        "text": texts[0],
        "page": metadatas[0].get("page", 0)
    }
)
```

### Metadata

Metadata helps us keep additional information associated with the vector.

Example:

```text
Vector
 ├── ID
 ├── Embedding
 └── Metadata
       ├── text
       └── page
```

---

# 🔹 Step 6: Convert User Question into Embedding

When the user asks a question, the question is also converted into an embedding using the same embedding model.

```python
query_embedding = model.encode([question])
```

This allows the question to be compared with document vectors.

---

# 🔹 Step 7: Similarity Search

The question embedding is searched against vectors stored in Pinecone.

```python
results = index.query(
    vector=query_embedding[0].tolist(),
    top_k=3,
    include_metadata=True
)
```

Pinecone returns the most relevant chunks.

Example:

```text
User Question
      ↓
Question Embedding
      ↓
Pinecone Similarity Search
      ↓
Top 3 Relevant Chunks
```

---

# 🔹 Step 8: Extract Retrieved Context

The text from the retrieved chunks is collected and used as context.

Example:

```python
context = "\n\n".join(
    match["metadata"]["text"]
    for match in results["matches"]
)
```

The retrieved information becomes the context for the LLM.

---

# 🔹 Step 9: Generate Final Answer using Groq

The retrieved context and user's question are passed to the LLM.

Example prompt:

```python
prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
```

The LLM then generates the final answer.

---

# 🔹 Complete RAG Flow

```text
                    OFFLINE / INDEXING
                    ==================

PDF
 ↓
PyPDFLoader
 ↓
Document Text
 ↓
RecursiveCharacterTextSplitter
 ↓
Chunks
 ↓
Sentence Transformer
 ↓
Embeddings
 ↓
Pinecone
 ↓
Vector Database


                    ONLINE / QUERY
                    ==============

User Question
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Pinecone Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Context
 ↓
Groq LLM
 ↓
Final Answer
```

---

# 🧠 Why RAG?

Traditional LLM:

```text
Question → LLM → Answer
```

RAG:

```text
Question
   ↓
Retrieve relevant information
   ↓
Provide information to LLM
   ↓
LLM generates answer
```

### Advantages of RAG

* Works with custom/private documents
* Provides relevant context to the LLM
* Reduces hallucination
* Can use frequently changing information
* No need to retrain the LLM for every document
* Useful for document question-answering systems

---

# 🔐 Security

API keys should **never be uploaded to GitHub**.

The following files should be included in `.gitignore`:

```gitignore
.env
myenv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

# 📦 Installation

Clone the project and create a virtual environment.

```bash
python -m venv myenv
```

Activate it on Windows:

```bash
myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

Do not commit this file to GitHub.

---

# ▶️ How to Run

1. Activate the virtual environment.

```bash
myenv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add API keys to `.env`.

4. Add your PDF/document inside the `data` folder.

5. Run the notebook or Python application.

```text
Load Document
      ↓
Split Document
      ↓
Create Embeddings
      ↓
Upload to Pinecone
      ↓
Ask Question
      ↓
Retrieve Relevant Chunks
      ↓
Generate Answer
```

---

# 📊 Example

### User Question

```text
What is the main purpose of this document?
```

### Retrieval

Pinecone retrieves the most relevant document chunks.

```text
Top Result 1 → Score: 0.82
Top Result 2 → Score: 0.76
Top Result 3 → Score: 0.69
```

### LLM

The retrieved chunks are provided to the Groq LLM.

### Final Output

```text
The document mainly explains ...
```

---

# 💡 Key Concepts Learned

Through this project, the following concepts are implemented:

* Document Loading
* PDF Text Extraction
* Document Splitting
* Chunk Size
* Chunk Overlap
* Embeddings
* Semantic Search
* Vector Database
* Pinecone
* Similarity Search
* Metadata
* Top-K Retrieval
* Context Retrieval
* Prompt Engineering
* Large Language Models
* Groq API
* Retrieval-Augmented Generation

---

# 🎤 Interview Explanation

### What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

It combines a retrieval system with a generative LLM.

First, relevant information is retrieved from a knowledge base using vector similarity search. Then the retrieved information is provided as context to an LLM, which generates the final answer.

### Why did you use Pinecone?

Pinecone is a managed vector database that allows us to efficiently store and search high-dimensional embeddings using similarity search.

### Why do we create embeddings?

Embeddings convert text into numerical vectors that capture semantic meaning, allowing us to find text that is semantically similar to a user's question.

### Why do we use chunking?

Large documents are divided into smaller meaningful pieces so that retrieval can identify the most relevant information instead of sending the complete document to the LLM.

### What happens when a user asks a question?

```text
Question
 ↓
Question Embedding
 ↓
Pinecone Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Context + Question
 ↓
LLM
 ↓
Answer
```

---

# 🚀 Future Improvements

Possible improvements for this project:

* Add Streamlit user interface
* Support multiple PDFs
* Add conversational memory
* Add source/page citations
* Implement hybrid search
* Add reranking
* Use LangChain/LangGraph for orchest
