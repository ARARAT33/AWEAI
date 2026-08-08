"""Example: start a chat with AWEAI."""

from aweai.models.inference import LLM

llm = LLM()  # auto-selects the best model for your hardware
print(llm.chat([
    {"role": "user", "content": "Hello! What can AWEAI do?"},
]))
