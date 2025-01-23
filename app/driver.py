import argparse

from app.core.config import Config
from app.core.aws_manager import AWSManager
from app.modules.bedrock_embedding import BedrockEmbeddings
from app.modules.pgvector_store import PGVectorStore
from app.modules.confluence_loader import ConfluenceDocumentLoader
from app.modules.bedrock_llm import BedrockLLM
from app.modules.markdown_recursive_splitter import MarkdownRecursiveChunking
from app.pipelines.rag_pipeline import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

def main(query: str = None):
    config = Config()
    aws_manager = config.aws_manager

    # Instantiate modules
    embeddings_module = BedrockEmbeddings(config, aws_manager)
    vector_store_module = PGVectorStore(config, embeddings_module, aws_manager)
    confluence_loader_module = ConfluenceDocumentLoader(config)
    llm_module = BedrockLLM(config, aws_manager)
    chunking_module = MarkdownRecursiveChunking(config)

    # Instantiate RAG pipeline
    rag_pipeline = RAGPipeline(
        config,
        confluence_loader_module,
        chunking_module,
        embeddings_module,
        vector_store_module,
        llm_module,
    )

    if query:
        # Generate response for a query
        response = rag_pipeline.generate_response(query)
        print(f"Response: {response}")
    else:
        # Run data ingestion
        rag_pipeline.ingest_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG application.")
    parser.add_argument(
        "-q", "--query", type=str, help="The user's query.", default=None
    )
    args = parser.parse_args()

    main(query=args.query)