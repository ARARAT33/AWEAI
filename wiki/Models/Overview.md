# Models

The Models category contains documentation for all model architectures, training utilities, and model management tools available in AWEAI.

## Subcategories

| Page | Description |
|------|-------------|
| [Catalog](Catalog.md) | Complete model type catalog and registry |
| [MLP](MLP.md) | Multi-Layer Perceptron |
| [CNN](CNN.md) | Convolutional Neural Network |
| [RNN](RNN.md) | Recurrent Neural Network |
| [LSTM](LSTM.md) | Long Short-Term Memory |
| [GRU](GRU.md) | Gated Recurrent Unit |
| [Transformer](Transformer.md) | Transformer architecture |
| [GAN](GAN.md) | Generative Adversarial Network |
| [Autoencoder](Autoencoder.md) | Autoencoder |
| [VAE](VAE.md) | Variational Autoencoder |
| [Diffusion](Diffusion.md) | Diffusion Model |
| [Linear](Linear.md) | Linear Regression |
| [Logistic](Logistic.md) | Logistic Regression |
| [NGram](NGram.md) | N-Gram Language Model |
| [KMeans](KMeans.md) | K-Means Clustering |
| [TimeSeries](TimeSeries.md) | Time Series Models |
| [Vision](Vision.md) | Vision Models |
| [Sequence](Sequence.md) | Sequence Models |
| [Inference](Inference.md) | Model Inference |
| [Registry](Registry.md) | Model Registry |
| [Selector](Selector.md) | Model Selection |
| [Trainer](Trainer.md) | Model Trainer |
| [APIs](APIs.md) | Model APIs |
| [Quantization](Quantization.md) | Model Quantization |
| [Export](Export.md) | Model Export |
| [Edge](Edge.md) | Edge Deployment |
| [FineTuning](FineTuning.md) | Fine-Tuning |
| [TransferLearning](TransferLearning.md) | Transfer Learning |
| [Pruning](Pruning.md) | Model Pruning |
| [Distillation](Distillation.md) | Knowledge Distillation |
| [ArchitectureSearch](ArchitectureSearch.md) | Neural Architecture Search |
| [HyperparameterTuning](HyperparameterTuning.md) | Hyperparameter Tuning |
| [AutoML](AutoML.md) | Automated Machine Learning |
| [ExperimentTracking](ExperimentTracking.md) | Experiment Tracking |
| [Serving](Serving.md) | Model Serving |
| [Monitoring](Monitoring.md) | Model Monitoring |
| [DriftDetection](DriftDetection.md) | Data Drift Detection |
| [Fairness](Fairness.md) | Model Fairness |
| [Explainability](Explainability.md) | Model Explainability |
| [Interpretability](Interpretability.md) | Model Interpretability |
| [AdversarialTesting](AdversarialTesting.md) | Adversarial Testing |
| [RedTeaming](RedTeaming.md) | Red Teaming |
| [Safety](Safety.md) | Model Safety |
| [Alignment](Alignment.md) | Model Alignment |
| [Guardrails](Guardrails.md) | Model Guardrails |

## Quick Start

```bash
# List available model types
aweai types

# Train a model
aweai train --type mlp --name my_model --data data.csv --target label

# Evaluate a model
aweai eval my_model --data test.csv --target label

# Export a model
aweai export my_model --fmt onnx
```

## Related Categories

- [Scale](../Scale/Overview.md) — Distributed training
- [Architecture](../Architecture/Overview.md) — Advanced architectures
- [Data](../Data/Overview.md) — Data engineering
- [Tools](../Tools/Overview.md) — Model tools
- [Commands](../Commands/Overview.md) — CLI commands
