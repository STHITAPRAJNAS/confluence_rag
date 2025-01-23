import unittest
from unittest.mock import patch, MagicMock

from app.core.config import Config
from app.modules.bedrock_embedding import BedrockEmbeddings
from app.modules.pgvector_store import PGVectorStore
from app.modules.confluence_loader import ConfluenceDocumentLoader
from app.modules.bedrock_llm import BedrockLLM
from app.modules.markdown_recursive_splitter import MarkdownRecursiveChunking
from app.pipelines.rag_pipeline import RAGPipeline
from app.utils.error_handler import ErrorHandler


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.aws_manager_mock = MagicMock()
        self.config.aws_manager = self.aws_manager_mock

        self.embeddings_mock = MagicMock(spec=BedrockEmbeddings)
        self.vector_store_mock = MagicMock(spec=PGVectorStore)
        self.document_loader_mock = MagicMock(spec=ConfluenceDocumentLoader)
        self.llm_mock = MagicMock(spec=BedrockLLM)
        self.chunking_mock = MagicMock(spec=MarkdownRecursiveChunking)
        self.error_handler_mock = MagicMock(spec=ErrorHandler)

        self.rag_pipeline = RAGPipeline(
            self.config,
            self.document_loader_mock,
            self.chunking_mock,
            self.embeddings_mock,
            self.vector_store_mock,
            self.llm_mock,
        )
        self.rag_pipeline.error_handler = self.error_handler_mock

    @patch('app.pipelines.rag_pipeline.logger')
    def test_ingest_data(self, mock_logger):
        documents = [{"content": "test content", "metadata": {"id": "1"}}]
        chunks = [{"page_content": "test chunk", "metadata": {"id": "1"}}]
        self.document_loader_mock.load.return_value = documents
        self.chunking_mock.chunk_document.return_value = chunks

        self.rag_pipeline.ingest_data()

        self.document_loader_mock.load.assert_called_once()
        self.chunking_mock.chunk_document.assert_called_once_with(documents[0])
        self.vector_store_mock.add_texts.assert_called_once_with(["test chunk"], [{"id": "1"}])
        mock_logger.info.assert_called()

    @patch('app.pipelines.rag_pipeline.logger')
    def test_ingest_data_no_documents(self, mock_logger):
        self.document_loader_mock.load.return_value = []

        self.rag_pipeline.ingest_data()

        self.document_loader_mock.load.assert_called_once()
        self.chunking_mock.chunk_document.assert_not_called()
        self.vector_store_mock.add_texts.assert_not_called()
        mock_logger.warning.assert_called_with("No documents loaded.")

    @patch('app.pipelines.rag_pipeline.logger')
    def test_generate_response(self, mock_logger):
        query = "test query"
        relevant_docs = [("test context",)]
        self.vector_store_mock.similarity_search.return_value = relevant_docs
        self.llm_mock.generate_text.return_value = "test answer"

        response = self.rag_pipeline.generate_response(query)

        self.vector_store_mock.similarity_search.assert_called_once_with(query, k=4)
        self.llm_mock.generate_text.assert_called_once_with(
            "Context:\ntest context\n\nQuestion:\ntest query\n\nAnswer:"
        )  # Using default prompt template
        self.assertEqual(response, "test answer")
        mock_logger.info.assert_called()

    def test_generate_response_error(self):
        query = "test query"
        self.vector_store_mock.similarity_search.side_effect = Exception("Test error")

        response = self.rag_pipeline.generate_response(query)

        self.error_handler_mock.handle_error.assert_called_once()
        self.assertEqual(response, "An error occurred while generating the response.")


if __name__ == "__main__":
    unittest.main()
