from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a document into smaller chunks.

        Args:
            document (Dict[str, Any]): The document to be chunked.

        Returns:
            List[Dict[str, Any]]: A list of document chunks.
        """
        if not isinstance(document, dict):
            raise TypeError("document must be a dictionary")
        if "page_content" not in document or not isinstance(document["page_content"], str):
            raise ValueError("document must contain a 'page_content' key with a string value")