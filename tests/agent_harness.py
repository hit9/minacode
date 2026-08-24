"""Shared harness for the agent test modules: an isolated session, a tool-call factory, and
the user-input queue helpers."""

from minacode.base import (
    ToolCall,
)
from minacode.config import (
    Config,
)
from minacode.session import Session, bootstrap_features


def session(tmp_path):
    # Isolate the data dir so tests never read the developer's real ~/.minacode (sessions, skills).
    config = Config()
    config.data_dir = str(tmp_path / "data")
    session = Session(cwd=str(tmp_path), config=config)
    bootstrap_features(session)
    return session


def call(name, args):
    return ToolCall(name + "-id", name, args)


def queue(s, *texts):
    for text in texts:
        s.enqueue_user_input(text)
