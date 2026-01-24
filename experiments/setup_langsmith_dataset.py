#!/usr/bin/env python
"""
Set up LangSmith Dataset for experiment evaluation.

This creates a well-defined dataset in LangSmith with:
- Multiple categories (JavaScript, Blockchain)
- Different difficulty levels (easy, medium, hard)
- Various question types (factual, conceptual, follow-up, comparison)

Run this ONCE to create the dataset, then use run_langsmith_experiment.py to run experiments.

Usage:
    python experiments/setup_langsmith_dataset.py
    python experiments/setup_langsmith_dataset.py --list
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

# Initialize LangSmith client
client = Client()

# Dataset name
DATASET_NAME = "rag-evaluation-tests-to-check-temperatures"

# ============================================================================
# COMPREHENSIVE TEST CASES
# ============================================================================
# Categories: javascript, blockchain
# Difficulty: easy, medium, hard
# Question Types: factual, conceptual, follow-up, comparison
# ============================================================================

TEST_CASES = [
    # =========================================================================
    # JAVASCRIPT - EASY (Factual questions with clear answers)
    # =========================================================================
    {
        "inputs": {"question": "What is JavaScript?"},
        "outputs": {
            "reference_answer": "JavaScript is a dynamic, lightweight, interpreted programming language commonly used for creating interactive web pages. It was first known as LiveScript and made its first appearance in Netscape 2.0 in 1995.",
            "expected_chunks": ["javascript_tutorial_parent_6"],
            "category": "javascript",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "js-001-what-is-js"
        }
    },
    {
        "inputs": {"question": "What are the advantages of JavaScript?"},
        "outputs": {
            "reference_answer": "JavaScript advantages include: less server interaction (validate input before sending to server), immediate feedback to visitors, increased interactivity with mouse/keyboard events, and richer interfaces with drag-and-drop and sliders.",
            "expected_chunks": ["javascript_tutorial_parent_8"],
            "category": "javascript",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "js-002-advantages"
        }
    },
    {
        "inputs": {"question": "What development tools can I use for JavaScript?"},
        "outputs": {
            "reference_answer": "JavaScript development tools include: simple text editors like Notepad, Microsoft FrontPage, Macromedia Dreamweaver MX, and Macromedia HomeSite 5. Since JavaScript is interpreted in the browser, you don't need a compiler.",
            "expected_chunks": ["javascript_tutorial_parent_10"],
            "category": "javascript",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "js-003-dev-tools"
        }
    },

    # =========================================================================
    # JAVASCRIPT - MEDIUM (Conceptual and syntax questions)
    # =========================================================================
    {
        "inputs": {"question": "How do I write my first JavaScript code?"},
        "outputs": {
            "reference_answer": "Place JavaScript code within <script>...</script> tags in HTML. Use document.write() to output text. Example: <script type='text/javascript'>document.write('Hello World!')</script>",
            "expected_chunks": ["javascript_tutorial_parent_12"],
            "category": "javascript",
            "difficulty": "medium",
            "question_type": "conceptual",
            "test_id": "js-004-first-code"
        }
    },
    {
        "inputs": {"question": "How do comments work in JavaScript?"},
        "outputs": {
            "reference_answer": "JavaScript supports single-line comments with // and multi-line comments with /* */. It also recognizes HTML comment opening <!-- but requires //-- for closing.",
            "expected_chunks": ["javascript_tutorial_parent_15"],
            "category": "javascript",
            "difficulty": "medium",
            "question_type": "conceptual",
            "test_id": "js-005-comments"
        }
    },
    {
        "inputs": {"question": "How do I enable JavaScript in Chrome?"},
        "outputs": {
            "reference_answer": "In Chrome: Click Chrome menu > Settings > Show advanced settings > Privacy section > Content settings > JavaScript section > Select 'Allow all sites to run JavaScript'.",
            "expected_chunks": ["javascript_tutorial_parent_18"],
            "category": "javascript",
            "difficulty": "medium",
            "question_type": "factual",
            "test_id": "js-006-enable-chrome"
        }
    },

    # =========================================================================
    # JAVASCRIPT - HARD (Follow-up and context-dependent questions)
    # =========================================================================
    {
        "inputs": {"question": "How do I install it?"},
        "outputs": {
            "reference_answer": "JavaScript doesn't require installation as it runs in web browsers. You only need a text editor to write code and a browser to run it.",
            "expected_chunks": ["javascript_tutorial_parent_10", "javascript_tutorial_parent_6"],
            "category": "javascript",
            "difficulty": "hard",
            "question_type": "follow-up",
            "test_id": "js-007-install-followup"
        }
    },
    {
        "inputs": {"question": "What's the difference between JavaScript and Java?"},
        "outputs": {
            "reference_answer": "JavaScript is a lightweight, interpreted language for web pages, while Java is a full programming language. JavaScript is complementary to and integrated with Java and HTML, but they are different languages.",
            "expected_chunks": ["javascript_tutorial_parent_6"],
            "category": "javascript",
            "difficulty": "hard",
            "question_type": "comparison",
            "test_id": "js-008-vs-java"
        }
    },

    # =========================================================================
    # BLOCKCHAIN - EASY (Basic definitions)
    # =========================================================================
    {
        "inputs": {"question": "What is blockchain?"},
        "outputs": {
            "reference_answer": "Blockchain is a tamper-proof, shared digital ledger that records transactions in a decentralized peer-to-peer network. It permanently stores the history of asset exchanges between participants in the network.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_5", "Blockchain_For_Beginners_A_EUBOF_Guide_parent_6"],
            "category": "blockchain",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "bc-001-what-is-blockchain"
        }
    },
    {
        "inputs": {"question": "What is Bitcoin?"},
        "outputs": {
            "reference_answer": "Bitcoin is the first and most renowned cryptocurrency, representing the first application of blockchain technology in financial services. It was created by Satoshi Nakamoto in 2009.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_5", "Blockchain_For_Beginners_A_EUBOF_Guide_parent_6"],
            "category": "blockchain",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "bc-002-what-is-bitcoin"
        }
    },
    {
        "inputs": {"question": "What is decentralization in blockchain?"},
        "outputs": {
            "reference_answer": "Decentralization eliminates the need for gatekeepers and single points of failure. It distributes power and control across a network, enhances security, and promotes transparency without central authority.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_5", "Blockchain_For_Beginners_A_EUBOF_Guide_parent_7"],
            "category": "blockchain",
            "difficulty": "easy",
            "question_type": "factual",
            "test_id": "bc-003-decentralization"
        }
    },

    # =========================================================================
    # BLOCKCHAIN - MEDIUM (Technical concepts)
    # =========================================================================
    {
        "inputs": {"question": "How does mining work in blockchain?"},
        "outputs": {
            "reference_answer": "Mining is the validation process where transactions are grouped into blocks. It takes computing effort to prove blocks, and once confirmed, they're easy to verify. Mining generates new bitcoins and builds trust by ensuring transactions are confirmed with significant computational work.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_5", "Blockchain_For_Beginners_A_EUBOF_Guide_parent_6"],
            "category": "blockchain",
            "difficulty": "medium",
            "question_type": "conceptual",
            "test_id": "bc-004-mining"
        }
    },
    {
        "inputs": {"question": "What are the different consensus mechanisms?"},
        "outputs": {
            "reference_answer": "Main consensus mechanisms include: Proof of Work (PoW) - miners solve puzzles, used by Bitcoin; Proof of Stake (PoS) - validators stake cryptocurrency; Delegated PoS (dPoS) - stakeholders vote for delegates; Proof of Authority (PoA) - approved validators, faster but less decentralized.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_7"],
            "category": "blockchain",
            "difficulty": "medium",
            "question_type": "factual",
            "test_id": "bc-005-consensus"
        }
    },
    {
        "inputs": {"question": "What is cryptography in blockchain?"},
        "outputs": {
            "reference_answer": "Blockchain cryptography includes digital signatures (authenticate transactions using private/public keys), hash functions (create unique fingerprints of data for integrity), and ensures security, integrity, and decentralization of the network.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_7"],
            "category": "blockchain",
            "difficulty": "medium",
            "question_type": "conceptual",
            "test_id": "bc-006-cryptography"
        }
    },
    {
        "inputs": {"question": "What are NFTs?"},
        "outputs": {
            "reference_answer": "NFTs (Non-Fungible Tokens) are unique digital assets that are non-interchangeable, cryptographically verified, and stored on blockchain. Key features include uniqueness, rarity, proof of ownership, immutability, and programmability.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_10"],
            "category": "blockchain",
            "difficulty": "medium",
            "question_type": "factual",
            "test_id": "bc-007-nfts"
        }
    },

    # =========================================================================
    # BLOCKCHAIN - HARD (Comparisons and complex topics)
    # =========================================================================
    {
        "inputs": {"question": "What is the difference between blockchain and traditional databases?"},
        "outputs": {
            "reference_answer": "Traditional databases are centralized with single entity control, easy data modification, and admin-controlled access. Blockchain is decentralized across many nodes, data is immutable once added, transparent to all participants, and uses cryptographic security.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_7"],
            "category": "blockchain",
            "difficulty": "hard",
            "question_type": "comparison",
            "test_id": "bc-008-vs-traditional-db"
        }
    },
    {
        "inputs": {"question": "What is DeFi and how is it different from traditional finance?"},
        "outputs": {
            "reference_answer": "DeFi (Decentralized Finance) operates on decentralized, peer-to-peer models through smart contracts, bypassing traditional intermediaries. Unlike TradFi which is centralized, custodial, and requires identity verification, DeFi is open, permissionless, and censorship-resistant.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_10"],
            "category": "blockchain",
            "difficulty": "hard",
            "question_type": "comparison",
            "test_id": "bc-009-defi"
        }
    },
    {
        "inputs": {"question": "What are CBDCs and how do they differ from cryptocurrencies?"},
        "outputs": {
            "reference_answer": "CBDCs (Central Bank Digital Currencies) are digital money issued by central banks, representing a claim against the issuing bank. Unlike decentralized cryptocurrencies like Bitcoin, CBDCs are centralized and controlled by central banks.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_10"],
            "category": "blockchain",
            "difficulty": "hard",
            "question_type": "comparison",
            "test_id": "bc-010-cbdc"
        }
    },
    {
        "inputs": {"question": "When should I use blockchain instead of a traditional database?"},
        "outputs": {
            "reference_answer": "Use blockchain when you need: immutable/tamper-proof records, decentralization without central authority, transparency and auditability, smart contracts, cryptographic security, cross-organizational trust, or cryptocurrency management.",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_7"],
            "category": "blockchain",
            "difficulty": "hard",
            "question_type": "conceptual",
            "test_id": "bc-011-when-to-use"
        }
    },

    # =========================================================================
    # EDGE CASES - Questions that test strict search behavior
    # =========================================================================
    {
        "inputs": {"question": "What is Python?"},
        "outputs": {
            "reference_answer": "NOT_IN_DOCUMENTS - The system should indicate that Python is not covered in the available documents.",
            "expected_chunks": [],
            "category": "edge-case",
            "difficulty": "hard",
            "question_type": "out-of-scope",
            "test_id": "edge-001-python"
        }
    },
    {
        "inputs": {"question": "Tell me about machine learning"},
        "outputs": {
            "reference_answer": "NOT_IN_DOCUMENTS - Machine learning is not covered in the available documents about JavaScript and Blockchain.",
            "expected_chunks": [],
            "category": "edge-case",
            "difficulty": "hard",
            "question_type": "out-of-scope",
            "test_id": "edge-002-ml"
        }
    },
]


def create_dataset():
    """Create or update the LangSmith dataset"""

    # Check if dataset exists
    existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing_datasets:
        print(f"⚠️  Dataset '{DATASET_NAME}' already exists.")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            client.delete_dataset(dataset_name=DATASET_NAME)
            print(f"🗑️  Deleted existing dataset")
        else:
            print("Keeping existing dataset. Exiting.")
            return

    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Comprehensive RAG evaluation dataset with JavaScript and Blockchain questions. Includes multiple difficulty levels and question types."
    )

    print(f"✅ Created dataset: {DATASET_NAME}")
    print(f"   ID: {dataset.id}")

    # Track statistics
    stats = {
        "category": {},
        "difficulty": {},
        "question_type": {}
    }

    # Add examples
    for i, test_case in enumerate(TEST_CASES, 1):
        outputs = test_case["outputs"]

        # Update stats
        for key in ["category", "difficulty", "question_type"]:
            val = outputs.get(key, "unknown")
            stats[key][val] = stats[key].get(val, 0) + 1

        client.create_example(
            inputs=test_case["inputs"],
            outputs=outputs,
            dataset_id=dataset.id,
            metadata={
                "category": outputs.get("category"),
                "difficulty": outputs.get("difficulty"),
                "question_type": outputs.get("question_type"),
                "test_id": outputs.get("test_id")
            }
        )
        print(f"   ✓ {i:2d}. [{outputs.get('difficulty', ''):6s}] {test_case['inputs']['question'][:50]}...")

    # Print summary
    print(f"\n{'='*60}")
    print(f"🎉 Dataset created with {len(TEST_CASES)} examples!")
    print(f"{'='*60}")

    print(f"\n📊 By Category:")
    for cat, count in sorted(stats["category"].items()):
        print(f"   {cat}: {count}")

    print(f"\n📈 By Difficulty:")
    for diff, count in sorted(stats["difficulty"].items()):
        print(f"   {diff}: {count}")

    print(f"\n❓ By Question Type:")
    for qtype, count in sorted(stats["question_type"].items()):
        print(f"   {qtype}: {count}")

    print(f"\n📋 View in LangSmith:")
    print(f"   https://smith.langchain.com/datasets")
    print(f"   Dataset: {DATASET_NAME}")


def list_dataset():
    """List examples in the dataset with detailed info"""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        examples = list(client.list_examples(dataset_id=dataset.id))

        print(f"\n📋 Dataset: {DATASET_NAME}")
        print(f"   ID: {dataset.id}")
        print(f"   Total Examples: {len(examples)}")
        print("\n" + "="*80)

        # Group by category
        by_category = {}
        for ex in examples:
            cat = ex.outputs.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ex)

        for category, exs in sorted(by_category.items()):
            print(f"\n🏷️  {category.upper()} ({len(exs)} questions)")
            print("-"*60)

            for ex in exs:
                diff = ex.outputs.get('difficulty', '?')
                qtype = ex.outputs.get('question_type', '?')
                test_id = ex.outputs.get('test_id', '?')
                question = ex.inputs.get('question', 'N/A')

                print(f"   [{diff:6s}] [{qtype:10s}] {question[:45]}...")
                print(f"            ID: {test_id}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Dataset '{DATASET_NAME}' may not exist. Run this script to create it.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List dataset examples")
    args = parser.parse_args()

    if args.list:
        list_dataset()
    else:
        create_dataset()
