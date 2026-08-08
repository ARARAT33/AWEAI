# Marketplace (v3.0)

Publish / download / rate models. Local-first registry at
`~/.aweai/market/index.json` with zip archives.

```bash
aweai market publish my_model --tag v1 --description "my first model"
aweai market search "mlp"
aweai market list
aweai market info <id>
aweai market download <id> --as my_copy
aweai market rate <id> 5
aweai market stats
```

## Python API

```python
from aweai.market import publish, search, download, rate, stats

publish("my_model", tag="v1", description="...")
download("mlp-20260808-1234", as_name="my_copy")
print(stats())
```

Publishing zips the model (model.json, version, quantized and edge
artifacts) so it can be shared across machines. Downloads are counted and
ratings are aggregated into `avg_rating`.
