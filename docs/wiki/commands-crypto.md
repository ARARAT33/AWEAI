# `aweai crypto` — utility commands

Group: **crypto** — 17 commands.

## `aweai crypto sha256`

**SHA-256 of text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello` | Input |

**Example:**

```bash
aweai crypto sha256 --text hello
```

## `aweai crypto sha512`

**SHA-512 of text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello` | Input |

**Example:**

```bash
aweai crypto sha512 --text hello
```

## `aweai crypto md5`

**MD5 of text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello` | Input |

**Example:**

```bash
aweai crypto md5 --text hello
```

## `aweai crypto sha1`

**SHA-1 of text.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | `hello` | Input |

**Example:**

```bash
aweai crypto sha1 --text hello
```

## `aweai crypto hmac`

**HMAC-SHA256.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--key` | `secret` | Key |
| `--text` | `message` | Message |

**Example:**

```bash
aweai crypto hmac --key secret
```

## `aweai crypto uuid`

**Generate UUID.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--version` | `4` | UUID version 1/4 |

**Example:**

```bash
aweai crypto uuid --version 4
```

## `aweai crypto uuid_many`

**Generate many UUIDs.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--count` | `5` | Count |

**Example:**

```bash
aweai crypto uuid_many --count 5
```

## `aweai crypto rand_int`

**Random integer in [lo, hi].**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--lo` | `1` | Low |
| `--hi` | `100` | High |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto rand_int --lo 1
```

## `aweai crypto rand_float`

**Random float in [0,1).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto rand_float --seed 1
```

## `aweai crypto rand_bytes`

**Random hex bytes.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--n` | `16` | Bytes |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto rand_bytes --n 16
```

## `aweai crypto rand_choice`

**Random choice from list.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--values` | `a,b,c` | Comma-separated |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto rand_choice --values a
```

## `aweai crypto rand_password`

**Random password.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--length` | `16` | Length |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto rand_password --length 16
```

## `aweai crypto xor`

**XOR bytes with key.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Plaintext |
| `--key` | `k` | Key |

**Example:**

```bash
aweai crypto xor --text hello
```

## `aweai crypto caesar`

**Caesar cipher shift.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Plaintext |
| `--shift` | `3` | Shift |

**Example:**

```bash
aweai crypto caesar --text hello
```

## `aweai crypto crc32`

**CRC32 checksum.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `hello` | Input |

**Example:**

```bash
aweai crypto crc32 --text hello
```

## `aweai crypto entropy`

**Estimate Shannon entropy (bits/char).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--text` | `aaaaaaaa` | Input |

**Example:**

```bash
aweai crypto entropy --text aaaaaaaa
```

## `aweai crypto token`

**URL-safe random token.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--bytes` | `32` | Bytes |
| `--seed` | `1` | Seed |

**Example:**

```bash
aweai crypto token --bytes 32
```
