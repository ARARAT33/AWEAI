# Evaluation

`aweai.eval` provides numpy-only metrics.

## Classification

- `accuracy(y_true, y_pred)`
- `precision(y_true, y_pred)`, `recall(...)`, `f1_score(...)`
- `confusion_matrix(y_true, y_pred)`
- `classification_report(y_true, y_pred)` — accuracy, per-class P/R/F1, macro

## Regression

- `mean_squared_error`, `mean_absolute_error`, `r2_score`

## Curves

- `loss_curve(history)` — training history
- `roc_curve_points(y_true, y_scores)` — ROC data
- `ascii_plot(values)` — terminal plot for loss curves
