from typing import List, Tuple, Dict, Any
from langchain_community.vectorstores.pgvector import PGVector as PostgresVectorStore
from app.core.vectorstore import VectorStore
from app.core.config import Config
from app.core.embeddings import Embeddings
from app.core.aws_manager import AWSManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PGVectorStore(VectorStore):
    def __init__(self, config: Config, embeddings: Embeddings, aws_manager: AWSManager):
        db_config = config.get_database_config()
        self.connection_string = f"postgresql://{db_config.get('user')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('dbname')}"
        self.collection_name = db_config.get("collection_name", "default_collection")
        self.embedder = embeddings
        self.rds_role_arn = db_config.get("assumed_role_arn")

        logger.info(f"Using PGVector store with connection string: {self.connection_string} and role ARN: {self.rds_role_arn}")

        # Get a new session for RDS, assuming a role if configured
        rds_session = aws_manager.assume_role(self.rds_role_arn) if self.rds_role_arn else aws_manager.get_session()

        # Modify the connection string to include the SSL mode
        self.connection_string = self.connection_string + "?sslmode=require"

        self.vector_store = PostgresVectorStore(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embedding_function=self.embedder,
            session=rds_session
        )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, embeddings: List[List[float]] = None, batch_size: int = 100) -> None:
        """
        Adds text, metadata and embeddings to the PGVectorStore in batches.

        Args:
            texts (List[str]): List of texts to add.
            metadatas (List[Dict[str, Any]], optional): List of metadata dictionaries. Defaults to None.
            embeddings (List[List[float]], optional): List of embeddings. Defaults to None.
            batch_size (int): The size of each batch. Defaults to 100.
        """
        super().add_texts(texts, metadatas)

        ids = [i for i in range(len(texts))]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
            batch_embeddings = embeddings[i:i + batch_size] if embeddings else None
            batch_ids = ids[i:i + batch_size]
            self.vector_store.add_embeddings_with_metadata(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        super().similarity_search(query, k)
        results = self.vector_store.similarity_search_with_score(query, k)
        return [(result.page_content, score) for result, score in results]