"""Example: natural-language automation studio.

Try:  "new model with this data"
      "index my documents"
      "run an agent"
"""

from aweai.actions.runner import ActionsRunner

runner = ActionsRunner(lang="en")
result = runner.run("new model with this data")
print(result)
