import faiss
import numpy as np


class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []

    def add_documents(self, documents, embeddings):
        embeddings = np.asarray(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents = documents

    def search(self, query_embedding, top_k=3):
        query_embedding = np.asarray(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index != -1:
                results.append({
                    "document": self.documents[index],
                    "score": float(score)
                })

        return results