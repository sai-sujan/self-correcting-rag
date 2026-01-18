import os
import glob
from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from config import MARKDOWN_DIR

def chunk_documents():
    """Split documents into parent and child chunks"""
    
    # Define how to split by headers
    headers_to_split = [
        ("#", "H1"),      # Main sections
        ("##", "H2"),     # Sub-sections  
        ("###", "H3")     # Sub-sub-sections
    ]
    
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )
    
    # Child splitter - fixed size chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 500 characters per chunk
        chunk_overlap=100    # 100 characters overlap
    )
    
    # Process all markdown files
    md_files = glob.glob(os.path.join(MARKDOWN_DIR, "*.md"))
    
    if not md_files:
        print(f"⚠️ No markdown files found in {MARKDOWN_DIR}/")
        return [], []
    
    all_parents = []
    all_children = []
    
    for md_path in md_files:
        print(f"📄 Processing: {Path(md_path).name}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Step 1: Split into parent chunks (by headers)
        parent_chunks = parent_splitter.split_text(text)
        
        # Step 2: Split each parent into children
        for i, parent in enumerate(parent_chunks):
            parent_id = f"{Path(md_path).stem}_parent_{i}"
            parent.metadata['parent_id'] = parent_id
            parent.metadata['source'] = Path(md_path).stem + ".pdf"
            
            all_parents.append((parent_id, parent))
            
            # Create children from this parent
            children = child_splitter.split_documents([parent])
            all_children.extend(children)
    
    print(f"\n✅ Created {len(all_parents)} parent chunks")
    print(f"✅ Created {len(all_children)} child chunks")
    
    return all_parents, all_children

if __name__ == "__main__":
    parents, children = chunk_documents()



"""
Real Example Workflow:
User asks: "How do I declare a variable in JavaScript?"
Step 1: Search children (small chunks)

Finds: "var, let, and const"

Step 2: Get parent ID from child

parent_id: "javascript_parent_5"

Step 3: Load full parent

Gets entire "Variables" section with examples

Step 4: LLM answers using full context

"In JavaScript, you can declare variables using var, let, or const. Here's the difference: [full explanation]"


Analogy:
Library with two systems:

Children = Index cards (quick lookup)
Parents = Full books (complete info)

You search index cards, but read the full book.
"""