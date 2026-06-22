#!/usr/bin/env python3
"""
FAQ RAG Chatbot - Main Entry Point

This application demonstrates RAG (Retrieval-Augmented Generation) using:
- Dummy FAQ data from XLSX
- Sentence Transformers for embeddings
- Chroma vector database for storage and retrieval
- Simple CLI interface for user interaction
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from vector_store import VectorStore
from retriever import Retriever
from cli import ChatbotCLI


def initialize_rag_system():
    """Initialize the RAG system components."""
    print("\n🚀 Initializing FAQ RAG System...\n")

    # 1. Load FAQ data from XLSX
    print("📂 Loading FAQ data...")
    data_loader = DataLoader("data/faq_data.xlsx")
    documents = data_loader.load()
    print(f"✓ Loaded {len(documents)} FAQ entries\n")

    # 2. Initialize vector store
    print("🗄️  Initializing vector store...")
    vector_store = VectorStore(db_path="./data/chroma_db")

    # Check if data already exists in vector store
    collection = vector_store.get_collection()
    existing_count = collection.count()

    if existing_count == 0:
        print("No existing data found. Indexing documents...")
        vector_store.add_documents(documents)
    else:
        print(f"✓ Found {existing_count} existing documents in vector store\n")

    # 3. Initialize retriever
    print("🔍 Initializing retriever...\n")
    retriever = Retriever(vector_store)

    return data_loader, vector_store, retriever


def main():
    """Main function to run the RAG chatbot."""
    try:
        # Initialize RAG system
        data_loader, vector_store, retriever = initialize_rag_system()

        # Initialize and run CLI
        cli = ChatbotCLI(retriever, vector_store)
        cli.run(data_loader)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
