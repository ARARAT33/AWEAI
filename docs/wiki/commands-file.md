# `aweai file` — utility commands

Group: **file** — 37 commands.

## `aweai file read`

**Read file content.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file read --path data.txt
```

## `aweai file write`

**Write text to file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `out.txt` | File path |
| `--text` | `hello` | Content |

**Example:**

```bash
aweai file write --path out.txt
```

## `aweai file append`

**Append text to file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | ---
| `--path` | `out.txt` | File path |
| `--text` | `more` | Content |

**Example:**

```bash
aweai file append --path out.txt
```

## `aweai file exists`

**Check file exists.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file exists --path data.txt
```

## `aweai file size`

**File size in bytes.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file size --path data.txt
```

## `aweai file lines`

**Count lines in file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file lines --path data.txt
```

## `aweai file words`

**Count words in file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file words --path data.txt
```

## `aweai file bytes`

**Count bytes in file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file bytes --path data.txt
```

## `aweai file type`

**File type by magic bytes.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file type --path data.txt
```

## `aweai file copy`

**Copy file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--src` | `a.txt` | Source |
| `--dst` | `b.txt` | Destination |

**Example:**

```bash
aweai file copy --src a.txt
```

## `aweai file move`

**Move file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--src` | `a.txt` | Source |
| `--dst` | `b.txt` | Destination |

**Example:**

```bash
aweai file move --src a.txt
```

## `aweai file delete`

**Delete file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `tmp.txt` | File path |

**Example:**

```bash
aweai file delete --path tmp.txt
```

## `aweai file touch`

**Create empty file / update mtime.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `new.txt` | File path |

**Example:**

```bash
aweai file touch --path new.txt
```

## `aweai file mkdir`

**Create directory (recursive).**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `dir/sub` | Directory path |

**Example:**

```bash
aweai file mkdir --path dir/sub
```

## `aweai file list`

**List directory entries.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Directory path |

**Example:**

```bash
aweai file list --path .
```

## `aweai file tree`

**Recursive file tree.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Directory path |
| `--depth` | `2` | Max depth |

**Example:**

```bash
aweai file tree --path .
```

## `aweai file glob`

**Find files by pattern.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--pattern` | `*.py` | Glob pattern |
| `--path` | `.` | Base dir |

**Example:**

```bash
aweai file glob --pattern *.py
```

## `aweai file find`

**Find files by name.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | `readme.md` | File name |
| `--path` | `.` | Base dir |

**Example:**

```bash
aweai file find --name readme.md
```

## `aweai file ext`

**File extension.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `archive.tar.gz` | File path |

**Example:**

```bash
aweai file ext --path archive.tar.gz
```

## `aweai file basename`

**Base name of path.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `/a/b/c.txt` | File path |

**Example:**

```bash
aweai file basename --path /a/b/c.txt
```

## `aweai file dirname`

**Directory of path.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `/a/b/c.txt` | File path |

**Example:**

```bash
aweai file dirname --path /a/b/c.txt
```

## `aweai file join_path`

**Join path parts.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--parts` | `a,b,c` | Comma-separated parts |

**Example:**

```bash
aweai file join_path --parts a
```

## `aweai file abs_path`

**Absolute path.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file abs_path --path data.txt
```

## `aweai file is_dir`

**Check directory.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Path |

**Example:**

```bash
aweai file is_dir --path .
```

## `aweai file is_file`

**Check file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | Path |

**Example:**

```bash
aweai file is_file --path data.txt
```

## `aweai file tail`

**Last n lines of file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |
| `--n` | `10` | Lines |

**Example:**

```bash
aweai file tail --path data.txt
```

## `aweai file head`

**First n lines of file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |
| `--n` | `10` | Lines |

**Example:**

```bash
aweai file head --path data.txt
```

## `aweai file grep`

**Search lines matching pattern.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |
| `--pattern` | `hello` | Regex |

**Example:**

```bash
aweai file grep --path data.txt
```

## `aweai file hash`

**SHA-256 of file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file hash --path data.txt
```

## `aweai file checksum`

**MD5 of file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file checksum --path data.txt
```

## `aweai file zip`

**Create zip archive.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `out.zip` | Zip path |
| `--sources` | `a.txt,b.txt` | Comma-separated files |

**Example:**

```bash
aweai file zip --path out.zip
```

## `aweai file unzip`

**Extract zip archive.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `out.zip` | Zip path |
| `--dest` | `extracted` | Destination dir |

**Example:**

```bash
aweai file unzip --path out.zip
```

## `aweai file gzip`

**Gzip-compress file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file gzip --path data.txt
```

## `aweai file gunzip`

**Gzip-decompress file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt.gz` | File path |

**Example:**

```bash
aweai file gunzip --path data.txt.gz
```

## `aweai file rename`

**Rename file.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--src` | `a.txt` | Old name |
| `--dst` | `b.txt` | New name |

**Example:**

```bash
aweai file rename --src a.txt
```

## `aweai file du`

**Directory size.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Directory path |

**Example:**

```bash
aweai file du --path .
```

## `aweai file mime`

**Guess MIME type.**

**Parameters:**

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `data.txt` | File path |

**Example:**

```bash
aweai file mime --path data.txt
```
