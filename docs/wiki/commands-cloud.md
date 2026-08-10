# `aweai cloud` — utility commands

Group: **cloud** — 7 commands.

## `aweai cloud weather`

**Weather for city (open-meteo, no key).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--city` | `Yerevan` | City name |
| `--lat` | `None` | Latitude (optional) |
| `--lon` | `None` | Longitude (optional) |

**Example:**

```bash
aweai cloud weather --city Yerevan
```

## `aweai cloud time`

**Time in timezone (worldtimeapi).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--zone` | `Asia/Yerevan` | IANA timezone |

**Example:**

```bash
aweai cloud time --zone Asia/Yerevan
```

## `aweai cloud geoip`

**GeoIP of current connection.**

## `aweai cloud ip`

**Public IP (alias).**

## `aweai cloud quote`

**Random quote (dummyjson).**

## `aweai cloud users`

**Random users (dummyjson).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--count` | `3` | Count |

**Example:**

```bash
aweai cloud users --count 3
```

## `aweai cloud cat_fact`

**Random cat fact.**
