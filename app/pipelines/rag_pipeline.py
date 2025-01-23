from app.core.config import Config
from app.core.document_loader import DocumentLoader
from app.core.chunking import ChunkingStrategy
from app.core.embeddings import Embeddings
from app.core.vectorstore import VectorStore
from app.core.llm import LLM
from app.utils.logger import get_logger
from app.utils.error_handler import ErrorHandler

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(
            self,
            config: Config,
            document_loader: DocumentLoader,
            chunking_strategy: ChunkingStrategy,
            embeddings: Embeddings,
            vector_store: VectorStore,
            llm: LLM,
    ):
        self.config = config
        self.document_loader = document_loader
        self.chunking_strategy = chunking_strategy
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.error_handler = ErrorHandler()

    def ingest_data(self):
        """
        Loads, chunks, and embeds documents, and adds them to the vector store.
        """
        try:
            logger.info("Starting data ingestion process...")
            documents = self.document_loader.load()
            if not documents:
                logger.warning("No documents loaded.")
                return

            all_chunks = []
            for doc in documents:
                chunks = self.chunking_strategy.chunk_document(doc)
                all_chunks.extend(chunks)

            texts = [chunk["page_content"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]

            logger.info(f"Embedding {len(texts)} chunks...")
            self.vector_store.add_texts(texts, metadatas)
            logger.info("Data ingestion completed successfully.")

        except Exception as e:
            self.error_handler.handle_error(e)

    def generate_response(self, query: str) -> str:
        """
        Generates a response to a query using the RAG pipeline.

        Args:
            query (str): The user's query.

        Returns:
            str: The generated response.
        """
        try:
            logger.info(f"Generating response for query: {query}")

            # 1. Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=4)
            context = "\n".join([doc[0] for doc in relevant_docs])

            # 2. Build the prompt
            prompt_template = self.config.get(
                "prompt_template", "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            )
            prompt = prompt_template.format(context=context, query=query)

            # 3. Generate the response
            response = self.llm.generate_text(prompt)

            logger.info("Response generated successfully.")
            return response

        except Exception as e:
            self.error_handler.handle_error(e)
            return "An error occurred while generating the response."