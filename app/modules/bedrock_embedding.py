from typing import List
from langchain_aws import embeddings as e
from app.core.embeddings import Embeddings
from app.core.config import Config
from app.core.aws_manager import AWSManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BedrockEmbeddings(Embeddings):
    def __init__(self, config: Config, aws_manager: AWSManager):
        embeddings_config = config.get_embeddings_config()
        self.model_id = embeddings_config.get("model_id", "amazon.titan-embed-text-v1")
        self.bedrock_role_arn = embeddings_config.get("assumed_role_arn")
        self.client = aws_manager.get_client("bedrock-runtime", assumed_role_arn=self.bedrock_role_arn)

        logger.info(f"Using Bedrock Embeddings with model ID: {self.model_id} and role ARN: {self.bedrock_role_arn}")

        # Verify imported library and class
        if not hasattr(e, "BedrockEmbeddings") or not isinstance(e.BedrockEmbeddings(client=self.client), Embeddings):
            raise ImportError("The imported 'BedrockEmbeddings' class is incorrect or not a subclass of 'Embeddings'.")

        self.embedder = e.BedrockEmbeddings(client=self.client, model_id=self.model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        super().embed_documents(texts)
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        super().embed_query(text)
        return self.embedder.embed_query(text)