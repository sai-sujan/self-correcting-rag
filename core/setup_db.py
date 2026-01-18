from qdrant_client.http import models as qmodels
from config import client, dense_embeddings, CHILD_COLLECTION

def setup_vector_db():
    """Create Qdrant collection for storing chunks"""
    
    # Get embedding size (how many numbers represent each chunk)
    embedding_size = len(dense_embeddings.embed_query("test"))
    print(f"📊 Embedding size: {embedding_size} dimensions")
    
    # Check if collection already exists
    if client.collection_exists(CHILD_COLLECTION):
        print(f"⚠️ Collection '{CHILD_COLLECTION}' already exists")
        choice = input("Delete and recreate? (y/n): ")
        if choice.lower() == 'y':
            client.delete_collection(CHILD_COLLECTION)
            print("🗑️ Deleted old collection")
        else:
            print("✓ Using existing collection")
            return
    
    # Create new collection
    client.create_collection(
        collection_name=CHILD_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=embedding_size,
            distance=qmodels.Distance.COSINE  # Measure similarity
        ),
        sparse_vectors_config={
            "langchain-sparse": qmodels.SparseVectorParams()  # For BM25
        }
    )
    
    print(f"✅ Created collection: {CHILD_COLLECTION}")

if __name__ == "__main__":
    setup_vector_db()



"""
## Explanation of `setup_db.py`

Let me break it down line by line:

---

### **1. Get Embedding Size**
```python
embedding_size = len(dense_embeddings.embed_query("test"))
```

**What's happening:**
- Takes the word "test" 
- Converts it to numbers (called "embedding")
- Counts how many numbers = 768

**Why:** Every chunk must be same size (768 numbers). Like all boxes must fit on the same shelf.

---

### **2. Check if Collection Exists**
```python
if client.collection_exists(CHILD_COLLECTION):
```

**What's a collection?**
- Think of it like a **table in a database**
- Stores all your document chunks
- Name: "document_child_chunks"

**Why check?** Don't want to accidentally delete existing data.

---

### **3. Create Collection**
```python
client.create_collection(
    collection_name=CHILD_COLLECTION,
    vectors_config=...
)
```

**This creates the storage container.**

---

### **4. Dense Vector Config**
```python
vectors_config=qmodels.VectorParams(
    size=embedding_size,  # 768 numbers per chunk
    distance=qmodels.Distance.COSINE  # How to measure similarity
)
```

**What's COSINE distance?**
- Measures how "similar" two chunks are
- Like asking: "Are these two sentences about the same thing?"
- Range: 0 (totally different) to 1 (identical)

**Example:**
- "I love pizza" vs "I enjoy pizza" = 0.95 (very similar)
- "I love pizza" vs "Quantum physics" = 0.1 (not similar)

---

### **5. Sparse Vector Config**
```python
sparse_vectors_config={
    "sparse": qmodels.SparseVectorParams()  # For BM25
}
```

**This enables keyword matching** (the BM25 we discussed earlier).

---

### **Big Picture:**

This file creates a **storage system** with TWO search methods:
1. **Dense vectors** = Semantic search (meaning-based)
2. **Sparse vectors** = Keyword search (exact matches)

**Analogy:**
- You're building a library
- Dense = Dewey Decimal System (organized by topic)
- Sparse = Index at the back (exact word lookup)

---

**Make sense?** Any questions before we continue?

"""