# `aweai audit` — utility commands

Group: **audit** — 6 commands.

## `aweai audit log`

**Append an entry to the audit log.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--action` | `file.write` | Action name |
| `--who` | `user` | Actor |
| `--detail` | `wrote config` | Detail |

**Example:**

```bash
aweai audit log --action file.write
```

## `aweai audit list`

**List recent audit log entries.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--limit` | `50` | Max entries |

**Example:**

```bash
aweai audit list --limit 50
```

## `aweai audit search`

**Search audit log by actor or action.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--query` | `file` | Search text |
| `--limit` | `20` | Max entries |

**Example:**

```bash
aweai audit search --query file
```

## `aweai audit clear`

**Clear the audit log.**

## `aweai audit permissions`

**Show role -> permission mapping.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--role` | `admin` | Role: admin|user|viewer |

**Example:**

```bash
aweai audit permissions --role admin
```

## `aweai audit compliance`

**Run a quick compliance checklist.**
