# `aweai backup` — utility commands

Group: **backup** — 5 commands.

## `aweai backup run`

**Back up a file/directory to the local backup store.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Path to back up |
| `--name` | `None` | Backup name (optional) |

**Example:**

```bash
aweai backup run --path .
```

## `aweai backup list`

**List backups.**

## `aweai backup restore`

**Restore a backup (copy back).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `backup-1` | Backup name |
| `--dest` | `.` | Destination |

**Example:**

```bash
aweai backup restore --name backup-1
```

## `aweai backup remove`

**Remove a backup.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `backup-1` | Backup name |

**Example:**

```bash
aweai backup remove --name backup-1
```

## `aweai backup version`

**Show backup versions (all names).**
