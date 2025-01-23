from typing import List, Dict, Any

from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from app.core.chunking import ChunkingStrategy
from app.core.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MarkdownRecursiveChunking(ChunkingStrategy):
    def __init__(self, config: Config):
        chunking_config = config.get("chunking", {})
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunking_config.get("markdown_chunk_size", 1000),
            chunk_overlap=chunking_config.get("markdown_chunk_overlap", 0),
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get("recursive_chunk_size", 200),
            chunk_overlap=chunking_config.get("recursive_chunk_overlap", 50),
        )

        logger.info("Initialized MarkdownRecursiveChunking strategy")

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        super().chunk_document(document)
        markdown_chunks = self.markdown_splitter.split_text(document["page_content"])
        chunks = []
        for markdown_chunk in markdown_chunks:
            recursive_chunks = self.recursive_splitter.split_text(markdown_chunk)
            for recursive_chunk in recursive_chunks:
                chunks.append(
                    {
                        "page_content": recursive_chunk,
                        "metadata": document["metadata"],  # Keep original metadata
                    }
                )
        return chunks