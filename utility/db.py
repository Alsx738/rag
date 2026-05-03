import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    """
    Returns a SQLAlchemy engine instance configured from environment variables.
    """
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_pass = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "postgres")

    postgres_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(postgres_url)

# Global reusable engine instance shared across the project
engine = get_engine()
