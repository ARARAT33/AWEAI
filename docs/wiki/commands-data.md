# data commands

Total commands: **11**

| Command | Description |
| --- | --- |
| `inspect` | Inspect the first rows of a dataset. |
| `split` | Split a JSONL dataset into train/valid/test. |
| `merge` | Merge multiple JSONL files. |
| `filter` | Filter rows by a field comparison. |
| `map` | Add a computed field to every row. |
| `normalize` | Normalize numeric columns (min-max or z-score). |
| `onehot` | One-hot encode a categorical field. |
| `tokenize` | Tokenize a text file. |
| `embed` | Create local embeddings (hash/tfidf) for documents. |
| `similarity` | Cosine similarity between two texts (hash embeddings). |
| `pipeline` | Run a declarative pipeline: [{"op":"filter","key":"age","op2":">","value":"18"}, {"op":"ma |
