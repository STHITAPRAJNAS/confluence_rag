from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Adds text and metadata to the VectorStore.

        Args:
            texts (List[str]): List of texts to add.
            metadatas (List[Dict[str, Any]], optional): List of metadata dictionaries. Defaults to None.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        if metadatas is not None and not isinstance(metadatas, list):
            raise TypeError("metadatas must be a list of dictionaries")
        if metadatas is not None and not all(isinstance(m, dict) for m in metadatas):
            raise TypeError("metadatas must be a list of dictionaries")

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """Performs a similarity search with a query.

        Args:
            query (str): The query string.
            k (int, optional): Number of results to return. Defaults to 4.

        Returns:
            List[Tuple[str, float]]: List of (text, score) tuples.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")