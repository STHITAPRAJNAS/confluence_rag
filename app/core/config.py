import yaml
import os
from dotenv import load_dotenv
from app.utils.logger import get_logger
from app.core.aws_manager import AWSManager

logger = get_logger(__name__)


class Config:
    def __init__(self, config_path="config/config.yaml"):
        load_dotenv()  # Load environment variables from .env file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.aws_manager = AWSManager(self)
        self.secret_manager = self.aws_manager.get_client("secretsmanager")

    def get(self, key, default=None):
        """
        Gets a configuration value.

        Priority:
        1. Environment variable (from .env or actual environment)
        2. Value in config.yaml
        3. Default value (if provided)

        Args:
            key (str): The configuration key.
            default: The default value if not found.

        Returns:
            The configuration value.
        """
        return os.environ.get(key.upper()) or self.config.get(key, default)

    def get_secret(self, secret_name, key=None):
        """
        Retrieves a secret from AWS Secrets Manager.

        Args:
            secret_name (str): The name of the secret in Secrets Manager.
            key (str, optional): A specific key within the secret to retrieve.

        Returns:
            The secret value (str or dict).
        """
        try:
            # Try to get from environment variable first
            env_secret_name = secret_name.upper().replace("-", "_")
            secret_value = os.environ.get(env_secret_name)

            if secret_value:
                if key:
                    try:
                        # Attempt to parse as JSON if a key is specified
                        import json

                        secret_dict = json.loads(secret_value)
                        return secret_dict.get(key)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse environment variable {env_secret_name} as JSON for key {key}. Returning raw value."
                        )
                        return secret_value
                else:
                    return secret_value

            # If not found in environment, try Secrets Manager
            get_secret_value_response = self.secret_manager.get_secret_value(
                SecretId=secret_name
            )

            if "SecretString" in get_secret_value_response:
                secret = get_secret_value_response["SecretString"]
                if key:
                    import json

                    secret_dict = json.loads(secret)
                    return secret_dict.get(key)
                else:
                    return secret
            else:
                logger.error(f"Secret {secret_name} is not a string.")
                return None

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None

    def get_database_config(self):
        db_config = self.config.get("database", {})
        return {
            "host": self.get("DATABASE_HOST", db_config.get("host")),
            "port": self.get("DATABASE_PORT", db_config.get("port")),
            "dbname": self.get("DATABASE_DBNAME", db_config.get("dbname")),
            "user": self.get("DATABASE_USER", db_config.get("user")),
            "password": self.get_secret(db_config.get("secret_name"), "password")
                        or self.get("DATABASE_PASSWORD"),  # Check secret manager first, then env
            "collection_name": self.get(
                "DATABASE_COLLECTION_NAME", db_config.get("collection_name")
            ),
            "assumed_role_arn": db_config.get("assumed_role_arn"),
        }

    def get_embeddings_config(self):
        embeddings_config = self.config.get("embeddings", {})
        return {
            "model_id": self.get("EMBEDDINGS_MODEL_ID", embeddings_config.get("model_id")),
            "assumed_role_arn": embeddings_config.get("assumed_role_arn")
        }

    def get_llm_config(self):
        llm_config = self.config.get("llm",{})
        return {
            "model_id": self.get("LLM_MODEL_ID", llm_config.get("model_id")),
            "model_kwargs": llm_config.get("model_kwargs",{}),
            "assumed_role_arn": llm_config.get("assumed_role_arn")
        }

    def get_confluence_config(self):
        confluence_config = self.config.get("confluence", {})
        return {
            "url": self.get("CONFLUENCE_URL", confluence_config.get("url")),
            "username": self.get_secret(confluence_config.get("secret_name"), "username")
                        or self.get(
                "CONFLUENCE_USERNAME"
            ),  # Check secret manager, then .env
            "api_key": self.get_secret(confluence_config.get("secret_name"), "api_key")
                       or self.get("CONFLUENCE_API_KEY"),  # Check secret manager, then .env
            "space_key": self.get("CONFLUENCE_SPACE_KEY", confluence_config.get("space_key")),
            "max_pages": self.get("CONFLUENCE_MAX_PAGES", confluence_config.get("max_pages")),
            "include_attachments": self.get(
                "CONFLUENCE_INCLUDE_ATTACHMENTS",
                confluence_config.get("include_attachments"),
            ),
            "limit": self.get("CONFLUENCE_LIMIT", confluence_config.get("limit")),
            "continue_on_failure": self.get(
                "CONFLUENCE_CONTINUE_ON_FAILURE",
                confluence_config.get("continue_on_failure"),
            ),
        }