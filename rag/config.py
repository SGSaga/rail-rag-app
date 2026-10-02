"""
Shared configuration for the rail RAG app.

Loads the OpenAI API key from the .env file and defines the constants
used across the other modules, so there is one place to change them.
"""

import os
from dotenv import load_dotenv

# Reads the .env file and puts OPENAI_API_KEY into the environment.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The small, cheap model. Good enough for this project, costs pennies.
LLM_MODEL = "gpt-4o-mini"

# The local embedding model. Small, runs on CPU, downloads once (~90MB).
EMBED_MODEL = "all-MiniLM-L6-v2"

# Where ChromaDB saves the vector store on disk.
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rail_incidents"

# Where the synthetic data lives.
DATA_PATH = "data/incidents.json"

# How many similar past incidents to retrieve for each query.
TOP_K = 3


def check_key():
    """Fail early with a clear message if the key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No OPENAI_API_KEY found. Copy .env.example to .env and paste your key in."
        )
