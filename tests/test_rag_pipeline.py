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
        documents = [{"page_content": "test content", "metadata": {"id": "1"}}]
        chunks = [{"page_content": "test chunk", "metadata": {"id": "1"}}]
        self.document_loader_mock.load.return_value = documents
        self.chunking_mock.chunk_document.return_value = chunks
        self.embeddings_mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        self.rag_pipeline.ingest_data()

        self.document_loader_mock.load.assert_called_once()
        self.chunking_mock.chunk_document.assert_called_once_with(documents[0])
        self.embeddings_mock.embed_documents.assert_called_once_with(["test chunk"])
        self.vector_store_mock.add_texts.assert_called_once_with(
            ["test chunk"], [{"id": "1"}]
        )
        mock_logger.info.assert_called()

    @patch("app.pipelines.rag_pipeline.logger")
    def test_ingest_data_no_documents(self, mock_logger):
        self.document_loader_mock.load.return_value = []

        self.rag_pipeline.ingest_data()

        self.document_loader_mock.load.assert_called_once()
        self.chunking_mock.chunk_document.assert_not_called()
        self.vector_store_mock.add_texts.assert_not_called()
        mock_logger.warning.assert_called_with("No documents loaded.")

    @patch("app.pipelines.rag_pipeline.logger")
    def test_generate_response(self, mock_logger):
        query = "test query"
        relevant_docs = [("test context", 0.8)]
        self.vector_store_mock.similarity_search.return_value = relevant_docs
        self.llm_mock.generate_text.return_value = "test answer"

        response = self.rag_pipeline.generate_response(query)

        self.vector_store_mock.similarity_search.assert_called_once_with(query, k=4)
        self.llm_mock.generate_text.assert_called_once()
        self.assertEqual(response, "test answer")
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