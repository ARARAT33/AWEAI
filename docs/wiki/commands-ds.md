# `aweai ds` — utility commands

Group: **ds** — 13 commands.

## `aweai ds normal`

**Sample n values from a normal distribution.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--mu` | `0.0` | Mean |
| `--sigma` | `1.0` | Std dev |
| `--n` | `5` | Count |

**Example:**

```bash
aweai ds normal --mu 0.0
```

## `aweai ds uniform`

**Sample n values from a uniform distribution.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--lo` | `0.0` | Low |
| `--hi` | `1.0` | High |
| `--n` | `5` | Count |

**Example:**

```bash
aweai ds uniform --lo 0.0
```

## `aweai ds poisson`

**Sample n values from a Poisson distribution.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--lam` | `3.0` | Lambda |
| `--n` | `5` | Count |

**Example:**

```bash
aweai ds poisson --lam 3.0
```

## `aweai ds binomial`

**Sample n values from a binomial distribution.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--trials` | `10` | Trials |
| `--prob` | `0.5` | Success prob |
| `--n` | `5` | Count |

**Example:**

```bash
aweai ds binomial --trials 10
```

## `aweai ds histogram`

**Compute histogram buckets.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,2,3,3,3,10` | Comma-separated numbers |
| `--bins` | `5` | Bin count |

**Example:**

```bash
aweai ds histogram --values 1
```

## `aweai ds quantile`

**Compute quantiles of a list.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5,6,7,8,9,10` | Comma-separated numbers |
| `--q` | `0.25,0.5,0.75` | Comma-separated quantiles |

**Example:**

```bash
aweai ds quantile --values 1
```

## `aweai ds skew`

**Sample skewness.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,2,3,3,3,10` | Comma-separated numbers |

**Example:**

```bash
aweai ds skew --values 1
```

## `aweai ds kurtosis`

**Sample excess kurtosis.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5` | Comma-separated numbers |

**Example:**

```bash
aweai ds kurtosis --values 1
```

## `aweai ds sample`

**Random sample without replacement.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5,6,7,8,9,10` | Comma-separated values |
| `--k` | `3` | Sample size |

**Example:**

```bash
aweai ds sample --values 1
```

## `aweai ds shuffle`

**Shuffle a list.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `a,b,c,d` | Comma-separated values |

**Example:**

```bash
aweai ds shuffle --values a
```

## `aweai ds standardize`

**Z-score standardize a number list.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5` | Comma-separated numbers |

**Example:**

```bash
aweai ds standardize --values 1
```

## `aweai ds normalize`

**Min-max normalize a number list to [0,1].**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--values` | `1,2,3,4,5` | Comma-separated numbers |

**Example:**

```bash
aweai ds normalize --values 1
```

## `aweai ds corr`

**Pearson correlation of two lists.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--x` | `1,2,3,4,5` | X values |
| `--y` | `2,4,5,4,5` | Y values |

**Example:**

```bash
aweai ds corr --x 1
```
