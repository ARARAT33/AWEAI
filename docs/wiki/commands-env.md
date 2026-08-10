# `aweai env` — utility commands

Group: **env** — 6 commands.

## `aweai env get`

**Get an environment variable.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `PATH` | Variable name |

**Example:**

```bash
aweai env get --name PATH
```

## `aweai env list`

**List environment variables (optionally filtered by prefix).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--prefix` | `` | Prefix filter |

## `aweai env load`

**Load a .env file and show parsed keys (without values).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.env` | .env path |

**Example:**

```bash
aweai env load --path .env
```

## `aweai env export`

**Export variables to a .env file (key=value pairs).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.env` | .env path |
| `--values` | `API_KEY=abc,DEBUG=true` | Comma-separated key=value |

**Example:**

```bash
aweai env export --path .env
```

## `aweai env unset`

**Check if an env var is set.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `API_KEY` | Variable name |

**Example:**

```bash
aweai env unset --name API_KEY
```

## `aweai env home`

**Show AWEAI home directory.**
