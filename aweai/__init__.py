"""AWEAI — the universal AI toolbox.

Everything AI in one lightweight Python package:

    pip install aweai
    aweai serve      # browser UI on http://localhost:8888
    aweai chat       # terminal chat
    aweai train ...  # create / fine-tune models
"""

from aweai.config import get_config, Config
from aweai.i18n import get_translator, LANGUAGES

__version__ = "2.0.0"
__all__ = [
    "__version__",
    "get_config",
    "Config",
    "get_translator",
    "LANGUAGES",
]
