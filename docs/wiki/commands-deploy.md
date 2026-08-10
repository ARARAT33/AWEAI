# `aweai deploy` — utility commands

Group: **deploy** — 10 commands.

## `aweai deploy plan`

**Show a deployment plan (steps).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--target` | `edge` | edge|cloud|desktop |
| `--app` | `aweai` | App name |

**Example:**

```bash
aweai deploy plan --target edge
```

## `aweai deploy package`

**Describe packaging steps for a target.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--target` | `wheel` | wheel|exe|appimage|apk|web |

**Example:**

```bash
aweai deploy package --target wheel
```

## `aweai deploy check`

**Check deploy prerequisites (python, git, etc.).**

## `aweai deploy push`

**Register a deployment (target + artifact).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--target` | `edge` | Target |
| `--artifact` | `model.onnx` | Artifact |

**Example:**

```bash
aweai deploy push --target edge
```

## `aweai deploy rollback`

**Roll back a deployment to previous version.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--deployment` | `edge-1` | Deployment id |

**Example:**

```bash
aweai deploy rollback --deployment edge-1
```

## `aweai deploy targets`

**List supported deployment targets.**

## `aweai deploy manifest`

**Generate a minimal deployment manifest.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `aweai-app` | App name |
| `--image` | `python:3.11` | Container image |
| `--port` | `8000` | Port |

**Example:**

```bash
aweai deploy manifest --name aweai-app
```

## `aweai deploy rollback`

**Plan a rollback strategy.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--current` | `v1.2` | Current version |
| `--previous` | `v1.1` | Previous version |

**Example:**

```bash
aweai deploy rollback --current v1.2
```

## `aweai deploy env-check`

**Check required env vars for deployment.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--required` | `API_KEY,DATABASE_URL` | Comma-separated required vars |

**Example:**

```bash
aweai deploy env-check --required API_KEY
```

## `aweai deploy compose`

**Generate docker-compose snippet.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `aweai-app` | Service name |
| `--image` | `python:3.11` | Image |
| `--port` | `8000` | Port |

**Example:**

```bash
aweai deploy compose --name aweai-app
```
