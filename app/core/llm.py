from abc import ABC, abstractmethod
from typing import Any

class LLM(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text based on a given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            str: The generated text.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")