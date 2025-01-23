import hashlib
from typing import Dict, List
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
        self.response_cache: Dict[str, str] = {}

    def ingest_data(self, batch_size: int = 100):
        """
        Loads, chunks, and embeds documents in batches, and adds them to the vector store.
        """
        try:
            logger.info("Starting data ingestion process...")

            offset = 0
            while True:
                logger.info(f"Loading documents with offset: {offset}")
                documents = self.document_loader.load(limit=batch_size, offset=offset)
                if not documents:
                    logger.warning("No more documents found.")
                    break

                all_chunks = []
                for doc in documents:
                    chunks = self.chunking_strategy.chunk_document(doc)
                    all_chunks.extend(chunks)

                if not all_chunks:
                    logger.warning("No chunks generated for this batch.")
                    offset += batch_size
                    continue

                texts = [chunk["page_content"] for chunk in all_chunks]
                metadatas = [chunk["metadata"] for chunk in all_chunks]

                logger.info(
                    f"Embedding and adding {len(texts)} chunks from batch {offset} to {offset + batch_size}..."
                )

                # Embed the documents and add them to the vector store
                embeddings = self.embeddings.embed_documents(texts)
                self.vector_store.add_texts(texts, metadatas=metadatas, embeddings=embeddings)

                offset += batch_size

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

            # 1. Check if the response is already cached
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            if query_hash in self.response_cache:
                logger.info("Returning cached response.")
                return self.response_cache[query_hash]

            # 2. Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=4)
            context = "\n".join([doc[0] for doc in relevant_docs])

            # 3. Build the prompt
            prompt_template = self.config.get(
                "prompt_template", "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            )
            prompt = prompt_template.format(context=context, query=query)

            # 4. Generate the response
            response = self.llm.generate_text(prompt)

            # 5. Cache the response
            self.response_cache[query_hash] = response

            logger.info("Response generated successfully.")
            return response

        except Exception as e:
            self.error_handler.handle_error(e)
            return "An error occurred while generating the response."