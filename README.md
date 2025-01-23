# LLM Confluence RAG Application

This project implements a Retrieval-Augmented Generation (RAG) application that leverages Large Language Models (LLMs) to answer questions based on content from a Confluence knowledge base. It uses a combination of powerful technologies, including:

*   **LangChain:** A framework for building applications with LLMs.
*   **Amazon Bedrock:** A fully managed service that makes high-performing foundation models from leading AI startups and Amazon available via a unified API.
*   **Amazon RDS for PostgreSQL with pgvector:** A managed relational database service with a vector extension for efficient similarity search.
*   **Confluence:** A collaborative workspace used as the source of knowledge for the application.
*   **AWS IAM Roles:** Securely manage access to AWS services for the application when deployed on EC2 or EKS.
*   **AWS Secrets Manager:** Securely store and retrieve sensitive information like API keys and database credentials.

## Features

*   **Data Ingestion:** Loads documents from a specified Confluence space, splits them into manageable chunks using a combination of Markdown and Recursive character splitting, and embeds them using Amazon Bedrock's Titan embedding model.
*   **Vector Storage:** Stores the document embeddings in an Amazon RDS for PostgreSQL database using the `pgvector` extension for efficient similarity search.
*   **Retrieval-Augmented Generation:** Retrieves relevant documents from the vector store based on a user's query and uses an LLM (via Amazon Bedrock) to generate a comprehensive and contextually relevant answer.
*   **Modular Design:** Uses abstract interfaces for core components (embeddings, vector store, document loader, LLM, chunking) and provides concrete implementations using specific technologies (Bedrock, PGVector, Confluence, etc.). This allows for flexibility and easy swapping of components.
*   **Configuration Management:** Uses a YAML configuration file (`config/config.yaml`) to manage application settings and environment-specific configurations.
*   **Secure Credential Handling:** Securely retrieves sensitive information like API keys and database credentials from either environment variables (for local development) or AWS Secrets Manager (for production).
*   **AWS Integration:** Designed to run seamlessly both locally (using AWS named profiles) and within AWS (using IAM roles). Automatically switches between profile-based and role-based authentication.
*   **Role Assumption:** Supports assuming specific IAM roles for accessing Bedrock and RDS, providing fine-grained access control.
*   **Error Handling and Logging:** Includes robust error handling and logging for debugging and monitoring.
*   **Dockerization:** Provides a `Dockerfile` for containerizing the application and a `docker-compose.yml` for local development.
*   **Production-Ready:** Designed with considerations for deployment to AWS (EC2 or EKS), scalability, and security.

## Project Structure

confluence-rag/
├── app/                     # Main application code
│   ├── core/                # Core abstractions and interfaces
│   │   ├── init.py
│   │   ├── config.py        # Configuration management
│   │   ├── embeddings.py    # Embeddings interface
│   │   ├── vectorstore.py   # Vector store interface
│   │   ├── document_loader.py # Document loader interface
│   │   ├── llm.py           # LLM interface
│   │   ├── chunking.py      # Chunking Strategy interface
│   │   └── aws_manager.py   # AWS Session Manager
│   ├── modules/             # Concrete implementations
│   │   ├── init.py
│   │   ├── bedrock_embedding.py
│   │   ├── pgvector_store.py
│   │   ├── confluence_loader.py
│   │   ├── bedrock_llm.py
│   │   └── markdown_recursive_splitter.py
│   ├── pipelines/           # Orchestration of components
│   │   ├── init.py
│   │   └── rag_pipeline.py
│   ├── utils/               # Utility functions (logging, error handling, etc.)
│   │   ├── init.py
│   │   ├── logger.py
│   │   └── error_handler.py
│   └── main.py              # Application entry point
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration
├── data/                    # Data storage (optional)
│   └── raw/
│   └── processed/
├── docs/                    # Documentation
│   └── index.md
├── tests/                   # Unit and integration tests
│   ├── init.py
│   ├── core/
│   ├── modules/
│   ├── pipelines/
│   └── utils/
├── .env.example              # Example environment variables file
├── requirements.txt         # Python dependencies
├── Dockerfile               # Dockerfile for containerization
├── docker-compose.yml       # Docker Compose configuration (local dev)
└── README.md                # Project documentation

## Prerequisites

*   **Python 3.11+**
*   **Docker** and **Docker Compose** (for local development)
*   **AWS Account:**
    *   An AWS account with appropriate permissions to create and manage resources (IAM, S3, RDS, Bedrock, Secrets Manager, EC2/EKS).
    *   Configure your AWS credentials locally for development. You can use named profiles in `~/.aws/credentials` and `~/.aws/config`.
*   **Confluence Account:**
    *   A Confluence instance that you have access to.
    *   A user account with appropriate permissions to read the content you want to ingest.
    *   An API key for your Confluence user.

## Setup

**1. Clone the Repository**

```bash
git clone <repository_url>
cd llm-confluence-rag

2. Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows

3. Install Dependencies
pip install -r requirements.txt

4. Configure the Application

config/config.yaml:
Update the database section with your RDS PostgreSQL connection details. If you are running locally, use localhost and the appropriate port (usually 5432). When deploying to AWS, use the RDS endpoint.
Update the embeddings and llm section with your desired Amazon Bedrock model IDs and assumed role ARNs, if needed. You can use the default models if you don't need to assume a specific role.
Update the confluence section with your Confluence URL, space key, and other relevant settings.
Create secrets in AWS Secrets Manager for your database password and Confluence credentials. The secret names should match what you put in config.yaml (e.g., prod/db_credentials, prod/confluence_credentials). The secrets should be in JSON format:
<!-- end list -->

# Example prod/db_credentials
{
  "password": "your_strong_db_password"
}

# Example prod/confluence_credentials
{
  "username": "your_confluence_username",
  "api_key": "your_confluence_api_key"
}

.env (for local development):
Create a .env file in the project root directory.
Add environment variables that correspond to your local settings (database, Confluence, etc.). You can use the .env.example file as a template. These will override values from config.yaml
Important: Do not commit your .env file to version control.
5. Run the Application (Local Development)

Using Docker Compose (Recommended):
<!-- end list -->

```bash
docker-compose up --build
```

This will build the Docker image, start the application container, and also start a PostgreSQL container for local development.

Directly using Python:

Make sure your virtual environment is activated.

Set the necessary environment variables (either in your shell or in a .env file).

Run the data ingestion:

<!-- end list -->

python app/main.py

python app/main.py -q "What is the policy on vacation days?"

6. Deployment to AWS (EC2 or EKS)

Database (RDS):
Create an Amazon RDS for PostgreSQL instance.
Enable the pgvector extension.
Configure the security group to allow access from your EC2 instance or EKS cluster.
Containerization:
Build the Docker image: docker build -t llm-confluence-rag .
Push the image to a container registry (e.g., Amazon ECR).
EC2:
Launch an EC2 instance.
Attach an IAM role that has permissions to access RDS, Bedrock, Secrets Manager, and assume other necessary roles (if applicable).
Deploy the Docker container to the EC2 instance. You might use user data scripts, Ansible, or other configuration management tools for this.
EKS:
Create an EKS cluster.
Create a Kubernetes deployment and service for your application using the Docker image you pushed to ECR.
Use IAM Roles for Service Accounts (IRSA) to grant your pods access to AWS services.
Configure a load balancer to expose your application.
CI/CD:
Set up a CI/CD pipeline (e.g., using Jenkins, GitLab CI, AWS CodePipeline) to automate the build, testing, and deployment process.
Usage
Once the application is running (either locally or deployed), you can interact with it by providing a query as a command-line argument:

Bash

python app/main.py -q "Your question here?"
The application will then:

Retrieve relevant documents from the Confluence knowledge base (via the vector store).
Construct a prompt using the retrieved documents and the user's query.
Send the prompt to the configured LLM (via Amazon Bedrock).
Return the LLM's generated response.
Testing
The project includes a tests/ directory for unit and integration tests. You can run the tests using pytest:

Bash

pytest tests/
Make sure you have the development dependencies installed (or a separate requirements-dev.txt if you created one):

Bash

pip install -r requirements-dev.txt # If you have a separate dev requirements file
