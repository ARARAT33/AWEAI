# `aweai api` — utility commands

Group: **api** — 6 commands.

## `aweai api check`

**Check if an API endpoint is reachable.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--url` | `https://api.github.com` | Endpoint URL |

**Example:**

```bash
aweai api check --url https://api.github.com
```

## `aweai api schema`

**Generate a JSON schema from field:type pairs.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `User` | Schema name |
| `--fields` | `id:int,name:string` | Comma-separated field:type |

**Example:**

```bash
aweai api schema --name User
```

## `aweai api curl`

**Build a curl command from parameters.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--method` | `GET` | HTTP method |
| `--url` | `https://api.example.com/data` | URL |
| `--data` | `` | JSON body (optional) |
| `--token` | `` | Bearer token (optional) |

**Example:**

```bash
aweai api curl --method GET
```

## `aweai api mock`

**Generate a mock JSON response.**

**Parameters:**

| Flag | Default | Description |
| `--fields` | `id:1,name:test` | Comma-separated field:value |

**Example:**

```bash
aweai api mock --fields id:1
```

## `aweai api endpoint-list`

**List common REST endpoints for a resource.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--resource` | `users` | Resource name |

**Example:**

```bash
aweai api endpoint-list --resource users
```

## `aweai api health`

**Build a health-check URL.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--base` | `https://api.example.com` | Base URL |
| `--path` | `/health` | Health path |

**Example:**

```bash
aweai api health --base https://api.example.com
```
