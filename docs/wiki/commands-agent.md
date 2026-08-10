# `aweai agent` — utility commands

Group: **agent** — 13 commands.

## `aweai agent create`

**Create an agent definition (role + system prompt + tools).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--role` | `general` | Role/purpose |
| `--system` | `You are a helpful assistant.` | System prompt |
| `--tools` | `search,memory,math` | Comma-separated tool list |
| `--model` | `local` | Preferred model |

**Example:**

```bash
aweai agent create --name assistant
```

## `aweai agent list`

**List defined agents.**

## `aweai agent get`

**Get an agent definition.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |

**Example:**

```bash
aweai agent get --name assistant
```

## `aweai agent remove`

**Remove an agent definition.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |

**Example:**

```bash
aweai agent remove --name assistant
```

## `aweai agent run`

**Run an agent on a task (simulated orchestration).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--task` | `summarize the input` | Task text |
| `--model` | `None` | Model override |

**Example:**

```bash
aweai agent run --name assistant
```

## `aweai agent chat`

**Multi-turn agent chat (memory-backed, simulated).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--message` | `Hello` | Message |

**Example:**

```bash
aweai agent chat --name assistant
```

## `aweai agent tools`

**List tools available to an agent.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |

**Example:**

```bash
aweai agent tools --name assistant
```

## `aweai agent grant`

**Grant a tool to an agent.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--tool` | `math` | Tool name |

**Example:**

```bash
aweai agent grant --name assistant
```

## `aweai agent revoke`

**Revoke a tool from an agent.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--tool` | `math` | Tool name |

**Example:**

```bash
aweai agent revoke --name assistant
```

## `aweai agent spawn`

**Spawn N worker agents for parallel tasks.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--role` | `worker` | Role |
| `--count` | `3` | Number of workers |
| `--task` | `process` | Base task |

**Example:**

```bash
aweai agent spawn --role worker
```

## `aweai agent multi`

**Multi-agent collaboration on one goal (simulated).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--goal` | `solve a problem` | Goal |
| `--agents` | `planner,critic,executor` | Agent names |

**Example:**

```bash
aweai agent multi --goal solve a problem
```

## `aweai agent system`

**Show the default system prompt template.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--role` | `assistant` | Role |

**Example:**

```bash
aweai agent system --role assistant
```

## `aweai agent register`

**Register an agent in the marketplace (local listing).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `assistant` | Agent name |
| `--description` | `General assistant` | Description |

**Example:**

```bash
aweai agent register --name assistant
```
