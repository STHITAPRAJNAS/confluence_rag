from abc import ABC, abstractmethod
from typing import List

class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text documents.

        Args:
            texts (List[str]): List of documents to embed

        Returns:
            List[List[float]]: List of embeddings (each embedding is a list of floats)
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string.

        Args:
            text (str): The query string.

        Returns:
            List[float]: The embedding of the query.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")