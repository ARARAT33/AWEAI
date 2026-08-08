"""UI entrypoint: `aweai serve` or `python -m aweai.ui`."""

from .api import serve

if __name__ == "__main__":
    serve()
