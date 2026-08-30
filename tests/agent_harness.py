"""Shared harness for the agent test modules: an isolated session, a tool-call factory, and
the user-input queue helpers."""

from wizolt.base import (
    ToolCall,
)
from wizolt.config import (
    Config,
    ProviderConfig,
)
from wizolt.session import Session, bootstrap_features


def session(tmp_path):
    # Isolate the data dir so tests never read the developer's real ~/.wizolt (sessions, skills).
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    bootstrap_features(session)
    return session


def session_with_provider(tmp_path, **provider_kwargs):
    """A session whose default provider is complete, so the real request path (e.g. compaction's
    provider resolution) can serve instead of tripping missing_fields."""
    s = session(tmp_path)
    provider_kwargs.setdefault("model", "gpt-4")
    provider_kwargs.setdefault("url", "http://test")
    provider_kwargs.setdefault("key", "sk-test")
    s.config.providers = {"default": ProviderConfig(**provider_kwargs)}
    return s


def call(name, args):
    return ToolCall(name + "-id", name, args)


def queue(s, *texts):
    for text in texts:
        s.enqueue_user_input(text)
