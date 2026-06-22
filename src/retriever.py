from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, vector_store, model_name: str = "all-MiniLM-L6-v2"):
        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant FAQ entries for a query."""
        # Get the collection
        collection = self.vector_store.get_collection()

        # Encode the query
        query_embedding = self.model.encode(query)

        # Query the vector store
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        retrieved_docs = []
        if results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                distances = results['distances'][0][i]
                # Convert cosine distance to similarity score (0-1)
                similarity = 1 - distances

                metadata = results['metadatas'][0][i]
                retrieved_docs.append({
                    'id': doc_id,
                    'question': metadata['question'],
                    'answer': metadata['answer'],
                    'category': metadata['category'],
                    'tags': metadata['tags'],
                    'similarity_score': similarity
                })

        return retrieved_docs

    def format_results(self, results: List[Dict]) -> str:
        """Format retrieval results for display."""
        if not results:
            return "No relevant FAQs found."

        output = []
        for i, doc in enumerate(results, 1):
            output.append(
                f"\n{'='*60}\n"
                f"Match #{i} (Relevance: {doc['similarity_score']:.2%})\n"
                f"{'='*60}\n"
                f"Category: {doc['category']}\n"
                f"Q: {doc['question']}\n"
                f"A: {doc['answer']}\n"
                f"Tags: {doc['tags']}"
            )
        return ''.join(output) + f"\n{'='*60}\n"
