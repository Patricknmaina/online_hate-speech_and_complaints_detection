import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add FastAPI directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FastAPI'))

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_tweet():
    """Sample tweet data for testing."""
    return {"text": "This is a test tweet for classification"}


@pytest.fixture
def sample_batch_tweets():
    """Sample batch of tweets for testing."""
    return [
        {"text": "First test tweet"},
        {"text": "Second test tweet"},
        {"text": "Third test tweet"}
    ]
