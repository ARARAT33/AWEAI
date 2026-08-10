# `aweai audio` — utility commands

Group: **audio** — 8 commands.

## `aweai audio info`

**Audio file metadata via stdlib (WAV) or ffprobe.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `sound.wav` | Audio path |

**Example:**

```bash
aweai audio info --path sound.wav
```

## `aweai audio tone`

**Generate WAV sine tone.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `tone.wav` | Output path |
| `--freq` | `440.0` | Hz |
| `--seconds` | `2.0` | Seconds |
| `--rate` | `22050` | Sample rate |

**Example:**

```bash
aweai audio tone --path tone.wav
```

## `aweai audio duration`

**Estimate audio duration from file size and bitrate.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--size_mb` | `4.0` | File size MB |
| `--bitrate_kbps` | `128.0` | Bitrate kbps |

**Example:**

```bash
aweai audio duration --size_mb 4.0
```

## `aweai audio nyquist`

**Nyquist frequency for a sample rate.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--sample_rate` | `44100` | Sample rate Hz |

**Example:**

```bash
aweai audio nyquist --sample_rate 44100
```

## `aweai audio sample-size`

**Bytes per second for PCM audio.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--sample_rate` | `44100` | Sample rate Hz |
| `--bits` | `16` | Bits per sample |
| `--channels` | `2` | Channels |

**Example:**

```bash
aweai audio sample-size --sample_rate 44100
```

## `aweai audio freq`

**Convert note name to frequency (A4=440).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--note` | `A4` | Note like A4, C5 |

**Example:**

```bash
aweai audio freq --note A4
```

## `aweai audio loudness`

**Approximate loudness (dB) from linear amplitude.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--amplitude` | `0.5` | Amplitude 0-1 |

**Example:**

```bash
aweai audio loudness --amplitude 0.5
```

## `aweai audio silence`

**Estimate silence duration in a clip.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--duration` | `120` | Clip seconds |
| `--silence_ratio` | `0.1` | Silence ratio 0-1 |

**Example:**

```bash
aweai audio silence --duration 120
```
