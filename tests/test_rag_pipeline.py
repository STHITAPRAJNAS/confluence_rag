import unittest
from unittest.mock import patch, MagicMock

from app.core.config import Config
from app.core.aws_manager import AWSManager
from app.modules.bedrock_embedding import BedrockEmbeddings
from app.modules.pgvector_store import PGVectorStore
from app.modules.confluence_loader import ConfluenceDocumentLoader
from app.modules.bedrock_llm import BedrockLLM
from app.modules.markdown_recursive_splitter import MarkdownRecursiveChunking
from app.pipelines.rag_pipeline import RAGPipeline
from app.utils.error_handler import ErrorHandler


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Mock the Config object
        self.config = MagicMock(spec=Config)
        self.config.get_database_config.return_value = {
            "host": "test_host",
            "port": 5432,
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "collection_name": "test_collection",
            "assumed_role_arn": "test_role_arn",
        }
        self.config.get_embeddings_config.return_value = {
            "model_id": "test_model_id",
            "assumed_role_arn": "test_role_arn",
        }
        self.config.get_llm_config.return_value = {
            "model_id": "test_model_id",
            "assumed_role_arn": "test_role_arn",
        }
        self.config.get_confluence_config.return_value = {
            "url": "test_url",
            "username": "test_username",
            "api_key": "test_api_key",
        }
        self.config.get.return_value = "test_value"
        self.config.get_secret.return_value = "test_secret"

        # Mock the AWSManager
        self.aws_manager_mock = MagicMock(spec=AWSManager)
        self.aws_manager_mock.get_client.return_value = MagicMock()
        self.aws_manager_mock.get_session.return_value = MagicMock()
        self.aws_manager_mock.assume_role.return_value = MagicMock()

        # Mock other modules
        self.embeddings_mock = MagicMock(spec=BedrockEmbeddings)
        self.vector_store_mock = MagicMock(spec=PGVectorStore)
        self.document_loader_mock = MagicMock(spec=ConfluenceDocumentLoader)
        self.llm_mock = MagicMock(spec=BedrockLLM)
        self.chunking_mock = MagicMock(spec=MarkdownRecursiveChunking)
        self.error_handler_mock = MagicMock(spec=ErrorHandler)

        # Instantiate RAGPipeline with mocks
        self.rag_pipeline = RAGPipeline(
            self.config,
            self.document_loader_mock,
            self.chunking_mock,
            self.embeddings_mock,
            self.vector_store_mock,
            self.llm_mock,
        )
        self.rag_pipeline.error_handler = self.error_handler_mock

    @patch("app.pipelines.rag_pipeline.logger")
    def test_ingest_data(self, mock_logger):
        # Simulate loading two batches of documents
        self.document_loader_mock.load.side_effect = [
            [{"page_content": "test content 1", "metadata": {"id": "1"}}],
            [{"page_content": "test content 2", "metadata": {"id": "2"}}],
            [],  # Empty list to signal the end of documents
        ]
        self.chunking_mock.chunk_document.side_effect = [
            [{"page_content": "test chunk 1", "metadata": {"id": "1"}}],
            [{"page_content": "test chunk 2", "metadata": {"id": "2"}}],
        ]
        self.embeddings_mock.embed_documents.side_effect = [
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]]
        ]

        self.rag_pipeline.ingest_data(batch_size=1)  # Ingest in batches of 1

        self.assertEqual(self.document_loader_mock.load.call_count, 3)  # Called 3 times (2 batches + 1 empty)
        self.chunking_mock.chunk_document.assert_any_call(
            {"page_content": "test content 1", "metadata": {"id": "1"}}
        )
        self.chunking_mock.chunk_document.assert_any_call(
            {"page_content": "test content 2", "metadata": {"id": "2"}}
        )
        self.embeddings_mock.embed_documents.assert_any_call(["test chunk 1"])
        self.embeddings_mock.embed_documents.assert_any_call(["test chunk 2"])
        self.vector_store_mock.add_texts.assert_any_call(
            ["test chunk 1"], metadatas=[{"id": "1"}], embeddings=[[0.1, 0.2, 0.3]]
        )
        self.vector_store_mock.add_texts.assert_any_call(
            ["test chunk 2"], metadatas=[{"id": "2"}], embeddings=[[0.4, 0.5, 0.6]]
        )
        mock_logger.info.assert_called()

    @patch("app.pipelines.rag_pipeline.logger")
    def test_ingest_data_no_documents(self, mock_logger):
        self.document_loader_mock.load.return_value = []

        self.rag_pipeline.ingest_data()

        self.document_loader_mock.load.assert_called_once_with(limit=100, offset=0)
        self.chunking_mock.chunk_document.assert_not_called()
        self.embeddings_mock.embed_documents.assert_not_called()
        self.vector_store_mock.add_texts.assert_not_called()
        mock_logger.warning.assert_called_with("No more documents found.")

    @patch("app.pipelines.rag_pipeline.logger")
    @patch('app.pipelines.rag_pipeline.hashlib.sha256')
    def test_generate_response(self, mock_hash, mock_logger):
        query = "test query"
        mock_hash.return_value.hexdigest.return_value = "test_hash"
        relevant_docs = [("test context", 0.8)]
        self.vector_store_mock.similarity_search.return_value = relevant_docs
        self.llm_mock.generate_text.return_value = "test answer"

        # Call generate_response twice with the same query
        response1 = self.rag_pipeline.generate_response(query)
        response2 = self.rag_pipeline.generate_response(query)

        self.vector_store_mock.similarity_search.assert_called_once_with(query, k=4)
        self.llm_mock.generate_text.assert_called_once()
        self.assertEqual(response1, "test answer")
        self.assertEqual(response2, "test answer") # Should return the cached response
        mock_logger.info.assert_called()

    def test_generate_response_error(self):
        query = "test query"
        self.vector_store_mock.similarity_search.side_effect = Exception(
            "Test error"
        )

        response = self.rag_pipeline.generate_response(query)

        self.error_handler_mock.handle_error.assert_called_once()
        self.assertEqual(response, "An error occurred while generating the response.")


if __name__ == "__main__":
    unittest.main()