# Vectorization

Vectorization converts data into numerical vectors for machine learning.

## Usage

```python
from aweai.data.vectorization import Vectorizer

vectorizer = Vectorizer(method="tfidf")
X_vec = vectorizer.fit_transform(texts)
```

## Related Pages

- [Embedding](Embedding.md) — Embeddings
- [FeatureEngineering](FeatureEngineering.md) — Feature engineering
