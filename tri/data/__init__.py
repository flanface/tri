"""Bundled game tables (JSON). `path(name)` is the file's location."""
import os


def path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
