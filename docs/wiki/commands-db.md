# `aweai db` — utility commands

Group: **db** — 9 commands.

## `aweai db create`

**Create SQLite table.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table name |
| `--schema` | `id INTEGER PRIMARY KEY, name TEXT` | Column schema |

**Example:**

```bash
aweai db create --path db.sqlite
```

## `aweai db insert`

**Insert row into SQLite.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |
| `--values` | `1,'hello'` | Values |

**Example:**

```bash
aweai db insert --path db.sqlite
```

## `aweai db query`

**Run SQL query.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--sql` | `SELECT * FROM items` | SQL |

**Example:**

```bash
aweai db query --path db.sqlite
```

## `aweai db tables`

**List SQLite tables.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |

**Example:**

```bash
aweai db tables --path db.sqlite
```

## `aweai db schema`

**Show table schema.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |

**Example:**

```bash
aweai db schema --path db.sqlite
```

## `aweai db count`

**Count rows in table.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |

**Example:**

```bash
aweai db count --path db.sqlite
```

## `aweai db drop`

**Drop table.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |

**Example:**

```bash
aweai db drop --path db.sqlite
```

## `aweai db delete`

**Delete rows.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |
| `--where` | `id=1` | WHERE clause |

**Example:**

```bash
aweai db delete --path db.sqlite
```

## `aweai db update`

**Update rows.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `db.sqlite` | DB path |
| `--table` | `items` | Table |
| `--set` | `name='x'` | SET clause |
| `--where` | `id=1` | WHERE clause |

**Example:**

```bash
aweai db update --path db.sqlite
```
