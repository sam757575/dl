from typing import Optional
import os

class ChatbotCLI:
    def __init__(self, retriever, vector_store):
        self.retriever = retriever
        self.vector_store = vector_store
        self.running = True

    def display_welcome(self):
        """Display welcome message."""
        print("\n" + "="*60)
        print("     FAQ RAG Chatbot - Powered by Vector Database")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question to search FAQs")
        print("  - 'reset'  : Clear vector database and reload data")
        print("  - 'exit'   : Exit the application")
        print("="*60 + "\n")

    def display_help(self):
        """Display help information."""
        print("\nAvailable commands:")
        print("  • Query: Type any question to search FAQs")
        print("  • reset: Clear and reload vector database")
        print("  • help : Show this message")
        print("  • exit : Exit the application\n")

    def reset_database(self, data_loader, db_path: str = "./data/chroma_db"):
        """Reset and reload the vector database."""
        print("\nResetting vector database...")

        # Clear existing collection
        self.vector_store.clear_collection()

        # Clear Chroma DB directory
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            print(f"✓ Cleared {db_path}")

        # Reinitialize vector store
        self.vector_store = type(self.vector_store)(db_path=db_path)
        self.vector_store.create_collection()

        # Reload documents
        docs = data_loader.load()
        self.vector_store.add_documents(docs)

        # Update retriever
        self.retriever.vector_store = self.vector_store

        print("✓ Vector database reset successfully\n")

    def process_query(self, query: str) -> None:
        """Process user query and display results."""
        if not query.strip():
            print("Please enter a valid query.\n")
            return

        print(f"\n🔍 Searching for: '{query}'\n")

        results = self.retriever.retrieve(query, top_k=3)
        formatted = self.retriever.format_results(results)
        print(formatted)

    def run(self, data_loader):
        """Run the interactive CLI loop."""
        self.display_welcome()

        while self.running:
            try:
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'exit':
                    print("\nThank you for using FAQ RAG Chatbot. Goodbye!\n")
                    self.running = False

                elif user_input.lower() == 'reset':
                    self.reset_database(data_loader)

                elif user_input.lower() == 'help':
                    self.display_help()

                else:
                    self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...\n")
                self.running = False
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
