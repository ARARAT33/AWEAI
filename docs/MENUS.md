# Megamenus, Terminal & Integrations (v3.0)

## Megamenus — 10,000+ commands & instructions

```bash
aweai allc                          # full catalog (10,000+ entries)
aweai allc --category quantization  # filter by category
aweai allc --search "edge"          # search
aweai allc --count 50               # cap rendered lines
aweai allc --json > catalog.json    # machine-readable output
aweai autoallc                      # 5,000+ automations
```

The catalog is generated deterministically from a base command table plus
combinatorial expansions over model types, data formats, tasks, export
formats, quantization formats, providers and languages. It is searchable,
categorized and can be scaled to millions of entries.

## In-app terminal

```bash
aweai terminal
```

A full REPL exposing every factory tool: `train`, `eval`, `models`,
`export`, `quantize`, `export-edge`, `dtrain`, `dworld`, `market ...`,
`integrations`, `rag ...`, `actions ...`, `allc`, `autoallc`, `autotest`,
`hardware`, `config`, `search`, `serve`, and more. Type `help` or `allc`
for the full catalog.

In the browser UI, the **Terminal** tab posts to `/api/terminal`.

## Integrations (BYOK)

```bash
aweai integrations list
aweai integrations chat --provider openai --message "hello"
```

Providers: `openai`, `google` (Gemini), `microsoft` (Azure OpenAI),
`anthropic` (Claude), `huggingface`. Keys are read from environment
variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `AZURE_OPENAI_KEY`,
`ANTHROPIC_API_KEY`, `HF_TOKEN`) or `~/.aweai/config.json`. No keys are
bundled; calls without keys return a helpful diagnostic.
