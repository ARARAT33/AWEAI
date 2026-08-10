# `aweai eval` — utility commands

Group: **eval** — 7 commands.

## `aweai eval accuracy`

**Accuracy of predictions vs labels (comma lists).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,0,1,1` | Predictions |
| `--label` | `1,0,0,1` | Labels |

**Example:**

```bash
aweai eval accuracy --pred 1
```

## `aweai eval precision`

**Precision score.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,0,1,1` | Predictions |
| `--label` | `1,0,0,1` | Labels |

**Example:**

```bash
aweai eval precision --pred 1
```

## `aweai eval recall`

**Recall score.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,0,1,1` | Predictions |
| `--label` | `1,0,0,1` | Labels |

**Example:**

```bash
aweai eval recall --pred 1
```

## `aweai eval f1`

**F1 score.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,0,1,1` | Predictions |
| `--label` | `1,0,0,1` | Labels |

**Example:**

```bash
aweai eval f1 --pred 1
```

## `aweai eval confusion`

**Confusion matrix counts.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,0,1,1` | Predictions |
| `--label` | `1,0,0,1` | Labels |

**Example:**

```bash
aweai eval confusion --pred 1
```

## `aweai eval mse`

**Mean squared error.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,2,3` | Predictions |
| `--label` | `1,3,3` | Labels |

**Example:**

```bash
aweai eval mse --pred 1
```

## `aweai eval mae`

**Mean absolute error.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pred` | `1,2,3` | Predictions |
| `--label` | `1,3,3` | Labels |

**Example:**

```bash
aweai eval mae --pred 1
```
