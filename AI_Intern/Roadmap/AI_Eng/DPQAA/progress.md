# RAG System — Design and Technology Report

## 1. Technology Stack

### 1.1 PDF Scanner: LlamaCloud

LlamaCloud is used to process PDF documents and convert them into Markdown. It also provides automatic OCR support for scanned documents.

#### Pros

* Free
* Easy to use
* Supports OCR operations

#### Cons

* No major disadvantages identified at the current stage

---

### 1.2 LLM: GPT-5 Nano

The system currently uses GPT-5 Nano as the primary language model.

#### Pros

* API access was provided for free

#### Cons

* Excessive usage could be costly for the team / Mr. Hung, so API usage should be kept reasonable

---

### 1.3 Embedding Model: `text-embedding-3-small`

The `text-embedding-3-small` model is used to convert document chunks and queries into vector representations for semantic retrieval.

#### Pros

* API access was provided for free
* Suitable for semantic search while keeping the embedding cost relatively low

#### Cons

* Excessive API usage should still be avoided

---

### 1.4 Indexing Strategy: Hierarchical Chunking

Documents are chunked according to their section hierarchy.

For example:

```text
Section 1
├── 1.1
│   ├── a.
│   ├── b.
│   └── c.
├── 1.2
│   └── a.
└── 1.3
```

The root section contains the complete content of all of its subsections. Subsections maintain their relationship with their parent/root section.

#### Pros

* Reduces semantic loss caused by splitting related information into independent chunks
* Supports both exact retrieval and full-context retrieval
* Preserves relationships between chunks
* Allows the system to retrieve a specific subsection while also recovering its broader context

#### Cons

* phuctapvail*n
* The hierarchy extraction is currently partially hard-coded
* More complicated to implement than conventional fixed-size chunking
* Potentially time-consuming during indexing and retrieval
* Some subsections may still contain thousands of tokens

#### Potential Improvement

If a subsection becomes too large (for example, more than 1,000 words), it can be further divided using fixed-size chunks with overlap.

This would:

* Prevent individual chunks from becoming excessively large
* Preserve semantic continuity through overlapping content
* Maintain the relationship between the smaller chunks and their original subsection
* Reduce the risk of diluting important information during retrieval

---

### 1.5 Vector Database: Qdrant Cloud

Qdrant Cloud is used as the vector database for storing document embeddings and performing vector retrieval.

#### Pros

* Free
* Easy to use
* Widely supported by existing vector-search frameworks
* Uses HNSW-based indexing for efficient vector search

#### Cons

* No major disadvantages identified at the current stage

---

## 2. Backend ↔ Frontend

### 2.1 Backend: FastAPI

FastAPI is used to expose the RAG system through an HTTP API.

#### Pros

* Easy to use
* Simple to integrate with the Python-based RAG backend

#### Cons

* Not as widely adopted as Flask

---

### 2.2 Frontend: Flutter

Flutter is used to build the user interface for the chatbot.

#### Pros

* Supports multiple platforms, including:

  * Web
  * Linux
  * iOS
  * Android
  * Other supported platforms

#### Cons

* Can be relatively complicated to develop compared with a simple web frontend

---

# 3. Design Choices

## 3.1 Why Conversation Routing?

Not every user query is self-contained. Some queries require information from previous messages in the conversation.

For example:

**Previous LLM answer:**

```text
This is the Python code for ...
```

```python
# Python code
...
```

**Current user query:**

```text
Explain line 10.
```

The current query does not contain enough information by itself to determine what "line 10" refers to.

Without conversation routing, the system may send the query directly to the query router or retrieval system, resulting in an incomplete or incorrect interpretation.

Conversation routing therefore determines whether the current query requires previous conversation context.

If context is required, the system generates a new, context-aware query using the conversation history before continuing with the normal routing process.

---

## 3.2 Why Query Routing?

Different types of queries require different instructions, tools, and backend operations.

Instead of sending every query through the same pipeline, the system first classifies the query and routes it to an appropriate workflow.

For example:

### FPT Policy Query

```text
"What are FPT's policies on child labour?"
```

Pipeline:

```text
Query
  ↓
FPT Query Route
  ↓
Qdrant Hybrid Search
  ↓
Build Context
  ↓
LLM
  ↓
Answer
```

### Coding Query

```text
"Generate Python code to do abcxyz."
```

Pipeline:

```text
Query
  ↓
Code Route
  ↓
Specialized Code Generation Model
  ↓
Generated Code
  ↓
Answer
```

A specialized coding model could be used for this route if necessary.

### Computational Query

```text
"What is the derivative of cosine with x = 5?"
```

Pipeline:

```text
Query
  ↓
Compute Route
  ↓
LLM Generates Python Code
  ↓
Execute Python Code
  ↓
Retrieve Result
  ↓
Answer
```

The important distinction is that the LLM does not need to perform the numerical computation itself. Instead, it generates executable code and the result is obtained by actually executing that code.

### General / Web Query

```text
"Who is the current president of Vietnam?"
```

Pipeline:

```text
Query
  ↓
General / Web Route
  ↓
Web Search
  ↓
Answer
```

This approach allows the system to use different tools and instructions depending on the user's intent.

---

## 3.3 Why Hybrid Search?

The system combines semantic search and BM25 keyword search.

### Semantic Search

Semantic search is good at understanding the meaning of a query.

However, it can sometimes miss documents containing important exact terms, especially when those terms are critical to identifying the correct document.

### BM25 Search

BM25 is effective at matching exact keywords and terms.

However, BM25 does not understand semantic meaning as well as embedding-based semantic search.

### Hybrid Search

Hybrid search combines both approaches:

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
      Semantic Search           BM25 Search
             │                     │
        Top 20 Results         Top 20 Results
             │                     │
             └──────────┬──────────┘
                        ↓
                   RRF Re-ranking
                        ↓
                    Top 5 Results
```

This allows the system to benefit from both:

* **Semantic understanding** from dense retrieval
* **Exact keyword matching** from BM25

The two result sets are then combined and re-ranked using Reciprocal Rank Fusion (RRF).

---

# 4. Future Improvements

## 4.1 Performance Optimization

The current system is relatively robust, but **slow AF**.

One of the major future goals is to optimize the system for speed.

Potential optimization areas include:

* Reducing unnecessary LLM calls
* Reducing redundant database queries
* Optimizing document traversal
* Optimizing embedding and retrieval operations
* Parallelizing independent operations
* Improving caching

One possible approach would be to implement computationally intensive critical sections in C or C++ if profiling shows that Python is actually becoming a bottleneck. (I'm obssessed with low-level programing).

However, this should only be considered after profiling the system. Rewriting components in C/C++ without identifying an actual bottleneck could increase complexity without providing meaningful performance improvements.

---

## 4.2 Improving Hybrid Search

Although hybrid search is significantly more robust than using only semantic search or BM25, it is still not perfect.

For example, a query such as:

```text
"FPT working overtime policies"
```

can sometimes fail to retrieve the correct document.

One possible reason is that the term **"FPT"** appears frequently across many documents. Therefore, it may provide little useful information for distinguishing between documents and can effectively become search noise.

Interestingly, a shorter query:

```text
"Working overtime policies"
```

can produce significantly better retrieval results.

### Short-Term Solution

Remove noisy terms such as `"FPT"` from the query before performing retrieval.

For example:

```text
Original:
"FPT working overtime policies"

After preprocessing:
"working overtime policies"
```

The disadvantage is that blindly removing terms could potentially damage the semantic meaning of the query.

### Long-Term Solution

Use the LLM as a **query re-router / query rewriter**.

Instead of manually removing specific words, the LLM can analyze the original query and generate a shorter, more precise search query.

For example:

```text
Original Query:
"FPT working overtime policies"

        ↓

LLM Query Rewriting

        ↓

Search Query:
"working overtime policies"

        ↓

Hybrid Search
```

This approach allows the system to preserve the user's original intent while producing a query that is more suitable for document retrieval.
