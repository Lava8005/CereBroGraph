"""Supabase Client Singleton.

Manages connection pooling and enforces environment variable validation
before allowing the pipeline to attempt database operations.
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError("Missing database dependencies. Run: pip install supabase python-dotenv")

# Force load the .env file from the project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def get_client() -> Client:
    """Initialize and return the Supabase client, enforcing strict auth checks."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "CRITICAL: Supabase credentials missing. "
            f"Ensure a .env file exists at {env_path} containing SUPABASE_URL and SUPABASE_KEY."
        )

    try:
        return create_client(url, key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Supabase client: {e}")

# Export a lazy-loaded instance for module imports
try:
    supabase = get_client()
except Exception as e:
    logging.error(str(e))
    supabase = None