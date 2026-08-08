# Data Pipeline

AWEAI loads, cleans, splits, normalizes, augments and tokenizes data.

## Loaders (`aweai.data.loaders`)

- `load_csv(path, target_column=..., feature_columns=...)`
- `load_json(path, text_key="text", label_key="label")`
- `load_jsonl(path, text_key="text", label_key="label")`
- `load_text(path)`
- `load_images(path)` — requires Pillow, returns flattened grayscale
- `load_any(path, ...)` — auto-detect by extension

## Split (`aweai.data.split`)

- `train_test_split(X, y, ratio=0.8, seed=...)`
- `split_by_ratio(n, ratio)`

## Normalize (`aweai.data.normalize`)

- `standardize(X)`, `minmax(X)`, `normalize_numeric(X, method)`
- `one_hot(labels)`, `label_encode(labels)`

## Augment (`aweai.data.augment`)

- `text_augment(text, n)` — shuffle/drop/repeat/swap
- `noise_augment(X, noise_std, n)`
- `image_augment_np(images, n, flip, shift, noise)`
- `augment(X=..., texts=..., images=...)`

## Own tokenizer (`aweai.data.tokenizer`)

No `tokenizers` dependency. Word-level tokenizer with special tokens,
`train`, `encode`, `decode`, `save`, `load`.
