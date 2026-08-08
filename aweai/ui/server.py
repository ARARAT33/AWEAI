"""Compatibility alias: create_app/serve live in api.py.""""

from .api import create_app, serve

__all__ = ["create_app", "serve"]
