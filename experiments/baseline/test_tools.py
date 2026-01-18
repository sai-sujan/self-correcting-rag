from tools import search_child_chunks, retrieve_parent_chunks

# Test 1: Search
print("🔍 Testing search...")
results = search_child_chunks.invoke({"query": "What is JavaScript?", "k": 3})
print(f"Found {len(results)} chunks\n")

if results:
    print("First result:")
    print(f"Content: {results[0]['content'][:100]}...")
    print(f"Parent ID: {results[0]['parent_id']}\n")
    
    # Test 2: Retrieve parent
    print("📖 Testing parent retrieval...")
    parent = retrieve_parent_chunks.invoke({"parent_ids": [results[0]['parent_id']]})
    print(f"Parent content length: {len(parent[0]['content'])} characters")