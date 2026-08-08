"""Example: RAG — index documents and ask questions."""

from pathlib import Path
import tempfile

from aweai.rag.engine import RAGEngine

docs = Path(tempfile.mkdtemp(prefix="aweai_rag_demo_")) / "docs.txt"
docs.write_text(
    "AWEAI is a universal AI toolbox created in Armenia.\n"
    "It supports RAG, agents, fine-tuning and a 12-language interface.\n"
    "The UI opens in the browser on port 8888.\n"
    "Yerevan, the capital of Armenia, is one of the oldest cities in the world.\n",
    encoding="utf-8",
)

engine = RAGEngine()
added = engine.index_file(str(docs))
print(f"Indexed {added} chunks")

result = engine.ask("What does AWEAI support?")
print("Answer:", result["answer"])
print("Sources:")
for s in result["sources"]:
    print("  -", s["id"], s["score"])
