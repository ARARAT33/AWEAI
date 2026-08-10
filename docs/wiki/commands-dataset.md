# `aweai dataset` — utility commands

Group: **dataset** — 18 commands.

## `aweai dataset stats`

**Dataset statistics (rows, columns, missing).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |

**Example:**

```bash
aweai dataset stats --path data.jsonl
```

## `aweai dataset split`

**Split a dataset into train/val/test.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |
| `--train` | `0.8` | Train ratio |
| `--val` | `0.1` | Val ratio |
| `--out` | `split` | Output prefix |

**Example:**

```bash
aweai dataset split --path data.jsonl
```

## `aweai dataset sample`

**Sample N rows from a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |
| `--n` | `10` | Rows |

**Example:**

```bash
aweai dataset sample --path data.jsonl
```

## `aweai dataset shuffle`

**Shuffle a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai dataset shuffle --path data.jsonl
```

## `aweai dataset dedupe`

**Remove duplicate rows from a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |

**Example:**

```bash
aweai dataset dedupe --path data.jsonl
```

## `aweai dataset merge`

**Merge two datasets.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--a` | `a.jsonl` | First path |
| `--b` | `b.jsonl` | Second path |
| `--out` | `merged.jsonl` | Output path |

**Example:**

```bash
aweai dataset merge --a a.jsonl
```

## `aweai dataset head`

**Show first N rows.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |
| `--n` | `5` | Rows |

**Example:**

```bash
aweai dataset head --path data.jsonl
```

## `aweai dataset version`

**Register a dataset version.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |
| `--version` | `v1` | Version tag |

**Example:**

```bash
aweai dataset version --path data.jsonl
```

## `aweai dataset validate`

**Validate dataset rows are dicts and non-empty.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.jsonl` | Path |

**Example:**

```bash
aweai dataset validate --path data.jsonl
```

## `aweai dataset create`

**Create a dataset definition from CSV data.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `my-ds` | Dataset name |
| `--data` | `a,b
1,2
3,4` | CSV data |

**Example:**

```bash
aweai dataset create --name my-ds
```

## `aweai dataset list`

**List datasets.**

## `aweai dataset show`

**Show dataset summary.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `my-ds` | Dataset name |

**Example:**

```bash
aweai dataset show --name my-ds
```

## `aweai dataset remove`

**Remove a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `my-ds` | Dataset name |

**Example:**

```bash
aweai dataset remove --name my-ds
```

## `aweai dataset split`

**Split CSV data into train/test by ratio.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--data` | `1,2
3,4
5,6
7,8
9,10` | CSV data |
| `--ratio` | `0.8` | Train ratio |

**Example:**

```bash
aweai dataset split --data 1
```

## `aweai dataset stats`

**Basic stats of a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--data` | `1,2,3,4,5,6,7,8,9,10` | CSV data |

**Example:**

```bash
aweai dataset stats --data 1
```

## `aweai dataset merge`

**Merge two CSV datasets vertically.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--a` | `h
1
2` | CSV A |
| `--b` | `h
3
4` | CSV B |

**Example:**

```bash
aweai dataset merge --a h
1
2
```

## `aweai dataset export`

**Export dataset to JSON file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `my-ds` | Dataset name |
| `--out` | `exports/dataset.json` | Output path |

**Example:**

```bash
aweai dataset export --name my-ds
```

## `aweai dataset version`

**Create a versioned snapshot of a dataset.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `my-ds` | Dataset name |
| `--version` | `v1` | Version tag |

**Example:**

```bash
aweai dataset version --name my-ds
```
