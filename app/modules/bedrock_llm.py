from typing import Any

from langchain_community.llms import Bedrock
from app.core.llm import LLM
from app.core.config import Config
from app.core.aws_manager import AWSManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BedrockLLM(LLM):
    def __init__(self, config: Config, aws_manager: AWSManager):
        llm_config = config.get_llm_config()
        self.model_id = llm_config.get("model_id", "anthropic.claude-v2")
        self.model_kwargs = llm_config.get("model_kwargs", {})
        self.bedrock_role_arn = llm_config.get("assumed_role_arn")
        self.client = aws_manager.get_client(
            "bedrock-runtime", assumed_role_arn=self.bedrock_role_arn
        )

        logger.info(f"Using Bedrock LLM with model ID: {self.model_id} and role ARN: {self.bedrock_role_arn}")

        self.llm = Bedrock(
            client=self.client, model_id=self.model_id, model_kwargs=self.model_kwargs
        )

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        super().generate_text(prompt, **kwargs)
        return self.llm.invoke(prompt, **kwargs)