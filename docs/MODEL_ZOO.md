# Model Zoo

All models are implemented **from scratch in numpy** (no Hugging Face).

| Type | Class | Task | Notes |
|------|-------|------|-------|
| `mlp` | `MLP` | classification / regression | SGD + backprop, softmax/MSE |
| `linear` | `LinearRegression` | regression | closed-form least squares |
| `logistic` | `LogisticRegression` | binary classification | SGD |
| `kmeans` | `KMeans` | clustering | Lloyd's algorithm |
| `ngram` | `NGramLM` | text generation | n-gram LM with fixed tuple-key serialization |
| `autoencoder` | `Autoencoder` | anomaly / embedding | undercomplete AE |
| `gan` | `GAN` | generative | MLP-based GAN |
| `rnn` | `RNN` | text / time-series | simple RNN |
| `lstm` | `LSTM` | text / time-series | LSTM cell |
| `cnn` | `TinyCNN` | vision | tiny CNN (numpy im2col) |
| `transformer` | `MiniTransformer` | text | mini decoder transformer |

Every model implements `fit`, `predict`, `state_dict`, `load_state`,
`save`, `load`, `export_json` — the factory interface.
