#!/usr/bin/env python3
"""
Demo script showing RAG system in action
Run: python3 demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from vector_store import VectorStore
from retriever import Retriever

def demo():
    print("\n" + "="*70)
    print("      FAQ RAG SYSTEM DEMO - Semantic Search in Action")
    print("="*70 + "\n")

    # Initialize
    print("📊 System Initialization:")
    print("  • Loading FAQ data from XLSX...")
    data_loader = DataLoader('data/faq_data.xlsx')
    documents = data_loader.load()
    print(f"  ✓ Loaded {len(documents)} FAQ entries\n")

    print("🗄️  Setting up vector store...")
    vector_store = VectorStore(db_path='./data/chroma_db')
    print(f"  ✓ Chroma DB ready\n")

    print("🔍 Initializing retriever...")
    retriever = Retriever(vector_store)
    print(f"  ✓ Retriever ready\n")

    # Demo queries
    demo_queries = [
        "How do I reset my password?",
        "What are the system requirements?",
        "Can I get a refund?",
        "How do I contact support?",
        "Is there a free trial?"
    ]

    print("="*70)
    print("RUNNING DEMO QUERIES")
    print("="*70)

    for query in demo_queries:
        print(f"\n{'─'*70}")
        print(f"📝 User Query: \"{query}\"")
        print(f"{'─'*70}")

        results = retriever.retrieve(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                score_bar = "█" * int(result['similarity_score'] * 20)
                print(f"\n🎯 Match #{i} ({result['similarity_score']:.1%}) {score_bar}")
                print(f"   Category: {result['category']}")
                print(f"   Q: {result['question']}")
                print(f"   A: {result['answer'][:100]}...")
        else:
            print("   ❌ No relevant FAQs found")

    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nTo use the interactive chatbot, run: python3 main.py\n")

if __name__ == "__main__":
    demo()
