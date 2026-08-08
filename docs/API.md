# AWEAI API Reference

## REST API

Base URL: `http://localhost:8888` (port auto-increments if busy).
OpenAPI: `http://localhost:8888/api/docs`.

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | `{"status": "ok", "version": "2.0.0"}` |
| GET | `/api/languages` | supported languages map |
| GET | `/api/config` | full config |
| POST | `/api/config` | body `{"values": {...}}` |

### Hardware & models
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/hardware` | detected resources + recommended tier |
| GET | `/api/models` | `{"catalog": [...], "installed": [...]}` |
| GET | `/api/models/recommended` | best model + suggestions for this machine |

### Chat
| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/api/chat` | `{"message": str, "history": [{"role","content"}], "model": ?}` | `{"reply", "model"}` |

### Training
| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/api/train` | `{"name", "data", "mode": "scrartch"\|"finetune"\|"continue", "base_model": ?, "epochs": 1}` | `{"status", "name", "path", "steps", "loss", "duration_s", "messages"}` |

### RAG
| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/api/rag/stats` | — | `{"backend","embedding","chunks","docs","index_file"}` |
| POST | `/api/rag/index` | `{"path": ?}` | `{"status","added","stats"}` |
| POST | `/api/rag/ask` | `{"query","top_k":4}` | `{"answer","sources":[{"id","text","score","metadata"}]}` |

### Agents
| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/api/agent/run` | `{"task","max_steps":5}` | `{"task","steps","final","tool_calls"}` |

### Actions
| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/api/actions/run` | `{"text","lang":"en"}` | intent result dict |

---

## Python API

```python
import aweai
from aweai.config import get_config
from aweai.hardware import detect
from aweai.models.selector import pick_best_model
from aweai.models.inference import LLM
from aweai.models.trainer import train_scratch, finetune, continue_training
from aweai.rag.engine import RAGEngine
from aweai.agents.engine import AgentEngine
from aweai.actions.runner import ActionsRunner
from aweai.i18n import get_translator

cfg = get_config()                 # JSON config
hw = detect()                      # hardware info dict
best = pick_best_model(hw)         # best model for this machine

llm = LLM()                        # auto backend + auto model
reply = llm.chat([{"role": "user", "content": "Hello"}])

result = train_scratch("m1", "data.jsonl", epochs=3)
result = finetune("Qwen/Qwen2.5-0.5B-Instruct", "tuned", "data.jsonl")
result = continue_training("m1", "~/.aweai/data/models/m1", "data.jsonl")

rag = RAGEngine()
rag.index_directory("docs/")
answer = rag.ask("What is AWEAI?")

agent = AgentEngine.create()
out = agent.run("calculate 6*7")

runner = ActionsRunner()
report = runner.run("new model with this data")

t = get_translator("hy")
print(t("welcome"))
```

---

## CLI reference

```
aweai [--version] {chat,serve,models,hardware,train,finetune,continue,rag,
                   agent,action,config,langs,doctor}
```

See `README.md` → Command line for full examples. `aweai doctor` prints
which optional stacks are installed.
