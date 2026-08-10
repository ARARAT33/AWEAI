# `aweai agent2` — utility commands

Group: **agent2** — 6 commands.

## `aweai agent2 spawn`

**Spawn a new agent run.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--agent` | `assistant` | Agent name |
| `--task` | `analyze data` | Task |

**Example:**

```bash
aweai agent2 spawn --agent assistant
```

## `aweai agent2 list-runs`

**List agent runs.**

## `aweai agent2 stop`

**Stop an agent run.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--run_id` | `abc123` | Run ID |

**Example:**

```bash
aweai agent2 stop --run_id abc123
```

## `aweai agent2 status`

**Show agent run status.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--run_id` | `abc123` | Run ID |

**Example:**

```bash
aweai agent2 status --run_id abc123
```

## `aweai agent2 scale`

**Scale agent replicas (simulated).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--agent` | `assistant` | Agent name |
| `--replicas` | `3` | Replica count |

**Example:**

```bash
aweai agent2 scale --agent assistant
```

## `aweai agent2 heartbeat`

**Record agent heartbeat.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--agent` | `assistant` | Agent name |

**Example:**

```bash
aweai agent2 heartbeat --agent assistant
```
