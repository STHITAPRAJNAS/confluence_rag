import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
from app.utils.logger import get_logger
from app.core.config import Config

logger = get_logger(__name__)


class AWSManager:
    def __init__(self, config: Config):
        self.config = config
        self.aws_profile = self.config.get("AWS_PROFILE")
        self.aws_region = self.config.get("AWS_REGION")
        self.session = self._create_session()

    def _create_session(self):
        """Creates a boto3 session, prioritizing named profiles for local development
        and falling back to IAM roles for AWS environments."""
        try:
            if self.aws_profile:
                # Use named profile for local development
                logger.info(f"Using AWS profile: {self.aws_profile}")
                session = boto3.Session(
                    profile_name=self.aws_profile, region_name=self.aws_region
                )
            else:
                # Rely on IAM roles in EC2/EKS or environment variables
                logger.info("Using default AWS session (IAM role or environment variables)")
                session = boto3.Session(region_name=self.aws_region)
            return session
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your credentials.")
            raise

    def get_session(self):
        """Returns the configured boto3 session."""
        return self.session

    def assume_role(self, role_arn: str, role_session_name: str = "AssumedSession"):
        """
        Assumes a specified IAM role and returns a new session.

        Args:
            role_arn (str): The ARN of the role to assume.
            role_session_name (str): The name to use for the assumed role session.

        Returns:
            boto3.Session: A new session using the assumed role credentials.
        """
        sts_client = self.get_client("sts")
        try:
            response = sts_client.assume_role(
                RoleArn=role_arn, RoleSessionName=role_session_name
            )

            credentials = response["Credentials"]
            assumed_role_session = boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
                region_name=self.aws_region,
            )
            logger.info(f"Successfully assumed role: {role_arn}")
            return assumed_role_session
        except ClientError as e:
            logger.error(f"Error assuming role {role_arn}: {e}")
            raise

    def get_client(self, service_name: str, assumed_role_arn: str = None):
        """
        Returns a boto3 client for the specified service. Optionally assumes a role before creating the client.

        Args:
            service_name (str): The name of the AWS service (e.g., 's3', 'bedrock', 'rds').
            assumed_role_arn (str, optional): The ARN of an IAM role to assume for this client.
        """
        if assumed_role_arn:
            session = self.assume_role(assumed_role_arn)
        else:
            session = self.session

        return session.client(service_name, region_name=self.aws_region)

    def get_resource(self, service_name: str, assumed_role_arn: str = None):
        """
        Returns a boto3 resource for the specified service. Optionally assumes a role before creating the resource.

        Args:
            service_name (str): The name of the AWS service (e.g., 's3', 'dynamodb').
            assumed_role_arn (str, optional): The ARN of an IAM role to assume for this resource.
        """
        if assumed_role_arn:
            session = self.assume_role(assumed_role_arn)
        else:
            session = self.session

        return session.resource(service_name, region_name=self.aws_region)