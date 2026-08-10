# `aweai feature` — utility commands

Group: **feature** — 16 commands.

## `aweai feature normalize`

**Min-max normalize a list of numbers.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4` | Values |

**Example:**

```bash
aweai feature normalize --values 1
```

## `aweai feature standardize`

**Z-score standardize a list of numbers.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4` | Values |

**Example:**

```bash
aweai feature standardize --values 1
```

## `aweai feature bin`

**Bin values into N buckets.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5,6` | Values |
| `--bins` | `3` | Bins |

**Example:**

```bash
aweai feature bin --values 1
```

## `aweai feature onehot`

**One-hot encode categorical values.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `cat,dog,cat,bird` | Values |

**Example:**

```bash
aweai feature onehot --values cat
```

## `aweai feature impute`

**Impute missing values (marked as 'NA') with strategy.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,NA,3,NA,5` | Values |
| `--strategy` | `mean` | mean|median|zero |

**Example:**

```bash
aweai feature impute --values 1
```

## `aweai feature select`

**Select top-k features by variance (heuristic).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--columns` | `a,b,c` | Column names |
| `--variance` | `0.5,0.1,0.9` | Variances |
| `--k` | `2` | Top K |

**Example:**

```bash
aweai feature select --columns a
```

## `aweai feature poly`

**Generate polynomial features for x.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--x` | `3.0` | Value |
| `--degree` | `3` | Degree |

**Example:**

```bash
aweai feature poly --x 3.0
```

## `aweai feature hash`

**Feature hashing: map token to bucket.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--token` | `hello` | Token |
| `--buckets` | `1000` | Buckets |

**Example:**

```bash
aweai feature hash --token hello
```

## `aweai feature bin`

**Bin numeric values into buckets.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5,6,7,8,9,10` | Comma-separated numbers |
| `--bins` | `5` | Bin count |

**Example:**

```bash
aweai feature bin --values 1
```

## `aweai feature onehot`

**One-hot encode labels.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--labels` | `cat,dog,cat,bird` | Comma-separated labels |

**Example:**

```bash
aweai feature onehot --labels cat
```

## `aweai feature scaling`

**Choose scaling method based on outliers.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,100` | Comma-separated numbers |

**Example:**

```bash
aweai feature scaling --values 1
```

## `aweai feature missing-fill`

**Suggest fill strategy for missing values.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--col_type` | `numeric` | numeric|categorical |

**Example:**

```bash
aweai feature missing-fill --col_type numeric
```

## `aweai feature select`

**Select top features by simple variance threshold.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,1,1,2,2,2,3,3,3,4` | Comma-separated numbers |
| `--threshold` | `0.5` | Variance threshold |

**Example:**

```bash
aweai feature select --values 1
```

## `aweai feature skew-fix`

**Suggest transform for skewed data.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--skew` | `2.5` | Skewness value |

**Example:**

```bash
aweai feature skew-fix --skew 2.5
```

## `aweai feature date-parts`

**Split ISO date into parts.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--date` | `2026-08-10` | ISO date |

**Example:**

```bash
aweai feature date-parts --date 2026-08-10
```

## `aweai feature text-length`

**Add text length feature suggestions.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello world` | Text |

**Example:**

```bash
aweai feature text-length --text hello world
```
