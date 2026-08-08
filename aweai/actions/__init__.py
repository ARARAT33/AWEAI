"""Actions subpackage: natural-language automation studio.

The Actions section (requirement #12) lets users say things like
"new model with this data" and AWEAI figures out the pipeline and runs it.
"""

from aweai.actions.runner import ActionsRunner, parse_action

__all__ = ["ActionsRunner", "parse_action"]
