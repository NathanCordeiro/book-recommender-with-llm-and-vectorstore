# 📚 Book Recommendation System with Vector Store, Embeddings, and LLMs

## 🧠 What is This Project?

This project is a smart **Book Recommendation System** that uses **vector embeddings**, **vector stores**, and **Large Language Models (LLMs)** to suggest similar books based on a given query or book description.

Instead of using traditional keyword matching or metadata-based filtering, it leverages **semantic understanding** through machine learning — allowing it to recommend books based on **meaning and context**, not just exact terms.

---

## ⚙️ How It Works

- **Data Loading**: A CSV file containing book metadata is loaded. This includes titles, descriptions, authors, and other attributes.

- **Text Embeddings Generation**: The book descriptions are converted into high-dimensional vector embeddings using an embedding model (like OpenAI or HuggingFace transformers).

- **Storing Embeddings in Vector Store**: These embeddings are saved in a vector database (such as FAISS or ChromaDB) that supports fast similarity searches.

- **Querying with LLM + Embedding**:
  - The user inputs a query (e.g., "Books similar to *The Alchemist*").
  - The query is embedded into the same vector space.
  - The vector store is searched for similar embeddings.
  - Matching books are returned as recommendations.

---

## 🧩 Key Concepts

### 🧾 Vector Store

A **vector store** is a special kind of database designed to store and search through **vector embeddings** efficiently. It supports fast **similarity search** (usually using cosine similarity or Euclidean distance).

Popular vector stores include:
- FAISS
- Chroma
- Pinecone

---

### 🧠 Embeddings

**Embeddings** are numerical representations of text that capture semantic meaning. In this project, book descriptions are turned into embeddings using a pre-trained model (e.g., Sentence Transformers, OpenAI Embeddings).

This allows the system to:
- Understand context
- Match similar content
- Go beyond keywords

---

### 🧮 How LLM Knows Similarity?

LLMs **don’t directly match text**, but when combined with embeddings, they can:
- Convert input queries into vectors
- Compare them with saved vectors in the database
- Retrieve the closest vectors based on **cosine similarity**

This way, the LLM can "understand" that two descriptions are semantically similar even if they don’t use the same words.


## 📂 Project Structure

```bash
main.ipynb           # Main Jupyter Notebook with all the logic
books.csv            # Dataset containing book metadata
README.md            # Documentation on project
index                # Contains embedded strings
```

## 🚀 How to Run
1. Install required libraries:

```bash
pip install pandas sentence-transformers faiss-cpu
```

2. Open `main.ipynb` in Jupyter Notebook.

3. Run all cells to:
- Load data
- Generate embeddings
- Save to vector store
- Perform search and view recommendations

> [!NOTE]
>
> I am using faiss-cpu but if you have a GPU install faiss-gpu
>
> The `index` embeddings file is not fully complete
>
> If you want the entire dataset to be embedded it will take a while  

## 🧠 FAQ & Real-World Considerations

### ❓ 1. Are LLMs practical for use as recommender systems in the industry, or are other deep learning methods like reinforcement learning (RL) more commonly applied?

While LLMs (Large Language Models) are increasingly used in recommendation systems — especially for semantic understanding and cold-start scenarios — they are **not yet the primary engine in large-scale, high-performance industrial recommenders**.

In production systems like Netflix, Amazon, or YouTube:

- **Traditional methods** (e.g., collaborative filtering, matrix factorization, deep learning models) still dominate.
- **Reinforcement Learning (RL)** is widely used for dynamic recommendations, especially in interactive environments (e.g., ad ranking, personalized feeds).
- **LLMs are more commonly used for:**
  - Context-aware recommendations (e.g., via chat)
  - Zero-shot reasoning
  - Summarizing item content
  - Enhancing explainability

✅ **Hybrid architectures** are popular — where LLMs are used alongside fast traditional models.

---

### ❓ 2. Extracting item embeddings with LLMs seems time-consuming. Does this make them less suitable for real-world use?

Yes, generating embeddings with LLMs **in real-time for every item** is not feasible due to:

- High latency  
- Expensive compute requirements

🚀 **Real-world approach**:

- Extract embeddings **offline in batch** (especially for static items like books or movies).
- Store them in a **vector database** (like FAISS, Chroma, or Pinecone).
- Only embed **dynamic queries** at runtime for fast similarity search.

This setup balances:

- ✅ Performance  
- ✅ Cost-effectiveness  
- ✅ Semantic power of LLMs
