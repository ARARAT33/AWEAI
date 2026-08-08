# RAG (numpy-only)

`aweai.rag` implements lightweight retrieval-augmented generation without
Hugging Face.

- Embeddings: hash bag-of-words vectors (numpy)
- Index: JSON on disk (`~/.aweai/data/rag/index.json`)
- Retrieval: cosine similarity

```python
from aweai.rag import RAGEngine
eng = RAGEngine()
eng.index_documents(["AWEAI is a model factory."])
eng.index_directory("docs/")
print(eng.ask("what is AWEAI?"))
```

**Bug fix**: the `index_file` shadowing bug (path vs dict sharing one
attribute) is fixed — the engine uses `index_path` for the file and `_index`
for the dict.
