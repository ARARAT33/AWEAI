# `aweai admin` — utility commands

Group: **admin** — 9 commands.

## `aweai admin uptime`

**System uptime (cross-platform).**

## `aweai admin users`

**List system users.**

## `aweai admin processes`

**Top processes by CPU.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--n` | `5` | Count |

**Example:**

```bash
aweai admin processes --n 5
```

## `aweai admin disk`

**Disk usage.**

## `aweai admin mem`

**Memory usage.**

## `aweai admin netstat`

**Network connections summary.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--port` | `0` | Filter by port (0 = all) |

**Example:**

```bash
aweai admin netstat --port 0
```

## `aweai admin cron-list`

**List cron jobs.**

## `aweai admin service`

**Check service status.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `docker` | Service name |

**Example:**

```bash
aweai admin service --name docker
```

## `aweai admin whoami`

**Current user and environment.**
