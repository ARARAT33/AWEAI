# `aweai dataops` — utility commands

Group: **dataops** — 6 commands.

## `aweai dataops pipeline-add`

**Add a data pipeline definition.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `etl` | Pipeline name |
| `--stages` | `extract,transform,load` | Comma-separated stages |

**Example:**

```bash
aweai dataops pipeline-add --name etl
```

## `aweai dataops pipeline-list`

**List data pipelines.**

## `aweai dataops pipeline-run`

**Simulate a pipeline run.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--name` | `etl` | Pipeline name |

**Example:**

```bash
aweai dataops pipeline-run --name etl
```

## `aweai dataops job-add`

**Add a data job.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--name` | `nightly` | Job name |
| `--cron` | `0 2 * * *` | Cron expression |
| `--action` | `pipeline-run etl` | Action |

**Example:**

```bash
aweai dataops job-add --name nightly
```

## `aweai dataops job-list`

**List data jobs.**

## `aweai dataops lineage`

**Build a simple lineage graph.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--source` | `raw.db` | Source |
| `--transform` | `clean.py` | Transform |
| `--sink` | `warehouse.db` | Sink |

**Example:**

```bash
aweai dataops lineage --source raw.db
```
