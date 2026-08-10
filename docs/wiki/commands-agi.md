# `aweai agi` — utility commands

Group: **agi** — 8 commands.

## `aweai agi score`

**Assess AGI readiness from 10 dimension scores (0-10 each).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--features` | `8,7,6,5,9,7,4,6,3,2` | Comma-separated scores: perception,reasoning,learning,planning,language,memory,creativity,social,embodiment,self_awareness |

**Example:**

```bash
aweai agi score --features 8
```

## `aweai agi level`

**Map a numeric score (0-10) to an AGI/ASI capability level.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--score` | `6.0` | Score 0-10 |

**Example:**

```bash
aweai agi level --score 6.0
```

## `aweai agi gaps`

**Identify weakest dimensions from a capability profile.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--features` | `8,7,6,5,9,7,4,6,3,2` | Comma-separated scores |

**Example:**

```bash
aweai agi gaps --features 8
```

## `aweai agi trajectory`

**Project capability growth given current score and yearly gain.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--score` | `6.0` | Current score 0-10 |
| `--gain` | `0.3` | Yearly gain |
| `--years` | `5` | Years ahead |

**Example:**

```bash
aweai agi trajectory --score 6.0
```

## `aweai agi alignment`

**List core alignment principles for safe AGI development.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--n` | `5` | Number of principles |

**Example:**

```bash
aweai agi alignment --n 5
```

## `aweai agi checklist`

**Return the AGI safety checklist.**

## `aweai agi capabilities`

**List known AGI capability areas.**

## `aweai agi self-improve-plan`

**Generate a recursive self-improvement plan.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--focus` | `reasoning` | Focus area |
| `--iterations` | `3` | Number of iterations |

**Example:**

```bash
aweai agi self-improve-plan --focus reasoning
```
