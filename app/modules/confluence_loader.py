from typing import List, Dict, Any

from langchain_confluence.confluence import ConfluenceLoader
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

    def load(self, **kwargs) -> List[Dict[str, Any]]:
        try:
            docs = self.loader.load(
                space_key=self.space_key,
                include_attachments=self.include_attachments,
                limit=self.limit,
                max_pages=self.max_pages,
                continue_on_failure=self.continue_on_failure,
                **kwargs,
            )
            documents = []
            for doc in docs:
                metadata = doc.metadata
                document = {"page_content": doc.page_content, "metadata": metadata}
                documents.append(document)
            return documents
        except Exception as e:
            logger.error(f"Error loading from Confluence: {e}")
            return []