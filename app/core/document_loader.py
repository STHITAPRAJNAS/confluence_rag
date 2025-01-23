from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        """Loads documents from the source.

        Returns:
            List[Dict[str, Any]]: List of documents (each document is a dictionary with text and metadata).
        """
        pass