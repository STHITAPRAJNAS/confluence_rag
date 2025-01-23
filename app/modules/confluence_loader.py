from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import ConfluenceLoader
from app.core.document_loader import DocumentLoader
from app.core.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ConfluenceDocumentLoader(DocumentLoader):
    def __init__(self, config: Config):
        confluence_config = config.get_confluence_config()
        self.url = confluence_config.get("url")
        self.username = confluence_config.get("username")
        self.api_key = confluence_config.get("api_key")
        self.loader = ConfluenceLoader(
            url=self.url, username=self.username, api_key=self.api_key
        )
        self.max_pages = confluence_config.get("max_pages", 100)
        self.space_key = confluence_config.get("space_key")
        self.include_attachments = confluence_config.get(
            "include_attachments", False
        )
        self.limit = confluence_config.get("limit", 50)
        self.continue_on_failure = confluence_config.get("continue_on_failure", True)

        logger.info(f"Initialized Confluence loader for space: {self.space_key} at URL: {self.url}")

    def load(self, limit: int = 50, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """
        Loads documents from Confluence with pagination support.

        Args:
            limit (int): The maximum number of documents to load per batch.
            offset (int): The starting offset for loading documents.
            **kwargs: Additional keyword arguments to pass to the ConfluenceLoader.

        Returns:
            List[Dict[str, Any]]: List of documents loaded from Confluence.
        """
        try:
            all_documents = []
            current_offset = offset
            while True:
                docs = self.loader.load(
                    space_key=self.space_key,
                    include_attachments=self.include_attachments,
                    limit=limit,
                    max_pages=self.max_pages,
                    continue_on_failure=self.continue_on_failure,
                    next_page_offset=current_offset,
                    **kwargs
                )
                if not docs:
                    break  # No more documents to load

                documents = []
                for doc in docs:
                    metadata = doc.metadata
                    document = {"page_content": doc.page_content, "metadata": metadata}
                    documents.append(document)
                all_documents.extend(documents)
                current_offset += limit  # Increment the offset for the next batch

            return all_documents
        except Exception as e:
            logger.error(f"Error loading from Confluence: {e}")
            return []