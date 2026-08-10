# `aweai crypto2` — utility commands

Group: **crypto2** — 12 commands.

## `aweai crypto2 base64-encode`

**Encode text to base64.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello world` | Text |

**Example:**

```bash
aweai crypto2 base64-encode --text hello world
```

## `aweai crypto2 base64-decode`

**Decode base64 to text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `aGVsbG8gd29ybGQ=` | Base64 text |

**Example:**

```bash
aweai crypto2 base64-decode --text aGVsbG8gd29ybGQ=
```

## `aweai crypto2 rot13`

**Apply ROT13 cipher.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Text |

**Example:**

```bash
aweai crypto2 rot13 --text hello
```

## `aweai crypto2 caesar`

**Apply Caesar cipher with shift.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Text |
| `--shift` | `3` | Shift |

**Example:**

```bash
aweai crypto2 caesar --text hello
```

## `aweai crypto2 url-encode`

**URL-encode a string.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `a b&c=d` | Text |

**Example:**

```bash
aweai crypto2 url-encode --text a b&c=d
```

## `aweai crypto2 url-decode`

**URL-decode a string.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `a%20b%26c%3Dd` | Encoded text |

**Example:**

```bash
aweai crypto2 url-decode --text a%20b%26c%3Dd
```

## `aweai crypto2 uuid`

**Generate UUIDs.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--n` | `1` | Count |
| `--version` | `4` | 4|1 |

**Example:**

```bash
aweai crypto2 uuid --n 1
```

## `aweai crypto2 random-hex`

**Generate random hex string.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--bytes` | `16` | Byte count |

**Example:**

```bash
aweai crypto2 random-hex --bytes 16
```

## `aweai crypto2 xor`

**XOR two hex strings.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--a` | `aabb` | Hex A |
| `--b` | `00ff` | Hex B |

**Example:**

```bash
aweai crypto2 xor --a aabb
```

## `aweai crypto2 checksum-text`

**Checksum of text (adler32).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Text |

**Example:**

```bash
aweai crypto2 checksum-text --text hello
```

## `aweai crypto2 compress`

**Compress text (zlib) to hex.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello hello hello hello` | Text |

**Example:**

```bash
aweai crypto2 compress --text hello hello hello hello
```

## `aweai crypto2 decompress`

**Decompress zlib hex to text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--hex` | `` | Compressed hex |
