"""Courtside web front end (FastAPI + Jinja2). See docs/WEB.md."""

from .app import create_app

__all__ = ["create_app"]
