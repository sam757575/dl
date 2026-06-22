import chromadb
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, db_path: str = "./data/chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)

        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def create_collection(self, collection_name: str = "faq") -> None:
        """Create or get a collection for FAQ data."""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents with embeddings to the vector store."""
        if self.collection is None:
            self.create_collection()

        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [
            {
                'question': doc['question'],
                'answer': doc['answer'],
                'category': doc['category'],
                'tags': doc['tags']
            }
            for doc in documents
        ]

        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        print("Adding documents to vector store...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        print(f"✓ Added {len(documents)} documents to vector store")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        if self.collection:
            # Get all IDs
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                print("✓ Vector store cleared")

    def get_collection(self):
        """Get the current collection."""
        if self.collection is None:
            self.create_collection()
        return self.collection
