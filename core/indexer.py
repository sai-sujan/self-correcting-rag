#Store Chunks in Database

import os
import json
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from config import (
    client, 
    dense_embeddings, 
    sparse_embeddings,
    CHILD_COLLECTION,
    PARENT_STORE_PATH
)
from chunker import chunk_documents

def index_documents():
    """Store parent chunks as JSON, child chunks in vector DB"""
    
    print("🔄 Starting document indexing...")
    
    # Get chunks from chunker
    parents, children = chunk_documents()
    
    if not children:
        print("❌ No chunks to index!")
        return
    
    # Step 1: Store child chunks in Qdrant (vector database)
    print(f"\n📊 Indexing {len(children)} child chunks into vector DB...")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID  # Use both dense + sparse
    )
    
    vector_store.add_documents(children)
    print("✅ Child chunks indexed!")
    
    # Step 2: Store parent chunks as JSON files
    print(f"\n💾 Saving {len(parents)} parent chunks to disk...")
    
    # Clear old parent files
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    for file in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, file))
    
    # Save each parent as separate JSON file
    for parent_id, parent_doc in parents:
        doc_dict = {
            "content": parent_doc.page_content,
            "metadata": parent_doc.metadata
        }
        
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    print("✅ Parent chunks saved!")
    print("\n🎉 Indexing complete!")

if __name__ == "__main__":
    index_documents()