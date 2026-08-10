# `aweai code` — utility commands

Group: **code** — 6 commands.

## `aweai code stats`

**Code statistics for a file (lines, comments, blank).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `aweai/cli.py` | File path |

**Example:**

```bash
aweai code stats --path aweai/cli.py
```

## `aweai code lint`

**Simple lint heuristics (line length, trailing spaces, tabs).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `aweai/cli.py` | File path |

**Example:**

```bash
aweai code lint --path aweai/cli.py
```

## `aweai code review`

**Generate a review checklist for a file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `aweai/cli.py` | File path |

**Example:**

```bash
aweai code review --path aweai/cli.py
```

## `aweai code todos`

**Find TODO/FIXME markers in a directory.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Directory |
| `--pattern` | `TODO|FIXME` | Regex |

**Example:**

```bash
aweai code todos --path .
```

## `aweai code format-json`

**Pretty-print a JSON file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.json` | Path |

**Example:**

```bash
aweai code format-json --path data.json
```

## `aweai code grep`

**Search text in files under a directory.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Directory |
| `--pattern` | `Copyright` | Regex |
| `--ext` | `py` | Extensions (comma) |

**Example:**

```bash
aweai code grep --path .
```
