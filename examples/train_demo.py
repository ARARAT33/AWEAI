"""Example: create a brand-new model from scratch.

This trains a tiny model on a small JSONL dataset so it runs on CPU in
seconds. For a real model, use a bigger dataset and `pip install aweai[ml]`.
"""

from pathlib import Path
import tempfile

from aweai.models.trainer import train_scratch

# tiny sample dataset
sample = Path(tempfile.mkdtemp(prefix="aweai_demo_")) / "data.jsonl"
sample.write_text(
    "\n".join([
        '{"text": "AWEAI is the universal AI toolbox."}',
        '{"text": "Բարեւ աշխարհ։ AWEAI-ը համընդհանուր AI գործիք է։"}',
        '{"text": "The capital of Armenia is Yerevan."}',
        '{"text": "RAG makes models answer from your documents."}',
        '{"text": "LoRA fine-tuning is fast and cheap."}',
    ]),
    encoding="utf-8",
)

result = train_scratch("demo_model", str(sample), epochs=2)
print(f"Trained model at: {result.path}")
print(f"Loss: {result.loss:.4f}, steps: {result.steps}")
print(f"Messages: {result.messages}")
