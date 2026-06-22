import pandas as pd
from typing import List, Dict

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self) -> List[Dict]:
        """Load FAQ data from XLSX file."""
        df = pd.read_excel(self.filepath)
        documents = []

        for _, row in df.iterrows():
            doc = {
                'id': str(row['ID']),
                'question': str(row['Question']),
                'answer': str(row['Answer']),
                'category': str(row['Category']),
                'tags': str(row['Tags']),
                # Combined text for embedding
                'text': f"{row['Question']} {row['Answer']}"
            }
            documents.append(doc)

        self.data = documents
        return documents

    def get_documents(self) -> List[Dict]:
        """Get loaded documents."""
        if self.data is None:
            self.load()
        return self.data
