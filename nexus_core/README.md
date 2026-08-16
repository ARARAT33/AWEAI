# NEXUS CORE - Advanced Autonomous AI System

## 🚀 Հզորագույն Ինքնավար ԱԻ Համակարգ

Nexus Core-ը առաջադեմ, ինքնավար արհեստական բանականության համակարգ է, որը կարող է **ինքնուրույն կատարել համակարգչային խնդիրներ**, սովորել փորձից և անընդհատ կատարելագործվել։

---

## ✨ Հիմնական Հնարավորություններ

### 🧠 Կոգնիտիվ Մշակում
- Բնական լեզվի հասկացողություն (NLP)
- Բազմաքայլ տրամաբանություն և եզրահանգումներ
- Համատեքստային որոշումների կայացում
- Մտադրությունների ճանաչում

### 🤖 Ավտոմատացում
- Ֆայլային համակարգի օպերացիաներ
- Վեբ ավտոմատացում (Selenium)
- Ծրագրերի կառավարում (PyAutoGUI)
- Համակարգային հրամաններ
- Պլանավորված խնդիրներ

### 📚 Ադապտիվ Ուսուցում
- Reinforcement Learning (Q-Learning)
- Online learning և փորձի կուտակում
- AutoML հիպերպարամետրերի օպտիմիզացիա
- Meta-learning մոտեցումներ

### 👁️ Multi-Modal Վերլուծություն
- Տեքստի վերլուծություն (sentiment, entities, topics)
- Նկարների ճանաչում (OpenCV)
- Աուդիո մշակում (librosa)
- Վիդեո անալիզ
- Կառուցվածքային տվյալների վերլուծություն

### 💾 Գիտելիքների Բազա
- Semantic search embeddings-ներով
- Ավտոմատ կատեգորիզացիա
- Forgetting mechanism (TTL)
- Experience storage և retrieval

---

## 🏗️ Ճարտարապետություն

```
nexus_core/
├── core/
│   └── engine.py              # Կենտրոնական շարժիչ
├── brain/
│   └── cognitive_processor.py # Տրամաբանություն և NLP
├── automation/
│   └── task_executor.py       # Ավտոմատացման շարժիչ
├── learning/
│   └── adaptive_learner.py    # Ուսուցման համակարգ
├── perception/
│   └── multi_modal_analyzer.py # Multi-modal վերլուծություն
├── memory/
│   └── knowledge_base.py      # Գիտելիքների կառավարում
├── utils/
│   ├── logger.py              # Logging utilities
│   └── config_manager.py      # Configuration management
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # Այս ֆայլը
```

---

## 🔧 Տեղադրում

### Պահանջվող Python տարբերակ
Python 3.8+

### Տեղադրել կախվածությունները

```bash
cd nexus_core
pip install -r requirements.txt
```

### Optional dependencies (առավելագույն հնարավորությունների համար)

```bash
# Computer Vision
pip install opencv-python pillow

# Advanced NLP
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm

# Web Automation
pip install selenium

# Desktop Automation
pip install pyautogui keyboard pygetwindow

# Audio Processing
pip install librosa
```

---

## 🎯 Օգտագործում

### Ինտերակտիվ ռեժիմ

```bash
python main.py --interactive
```

### Մեկ խնդիր կատարել

```bash
python main.py --task "Անալիզ արա իմ documents թղթապանակը"
```

### Կոնֆիգուրացիայով

```bash
python main.py --config config.yaml
```

---

## 📋 Օրինակներ

### 1. Ֆայլային օպերացիաներ

```python
from nexus_core.core.engine import NexusEngine

engine = NexusEngine()
await engine.initialize_components()
await engine.start()

# Ֆայլ ստեղծել
task = {
    'type': 'automation',
    'action': 'file_operation',
    'parameters': {
        'operation': 'create',
        'source': './test_file.txt',
        'content': 'Hello from Nexus AI!'
    }
}

await engine.submit_task(task)
```

### 2. Տեքստի վերլուծություն

```python
# Տեքստի անալիզ
analysis_task = {
    'type': 'analysis',
    'modality': 'text',
    'data': 'This is an amazing product with great features!'
}

result = await engine.multi_modal_analyzer.analyze(analysis_task)
print(result['sentiment'])  # {'positive': 0.8, ...}
```

### 3. Բնական լեզվի հարցեր

```python
# Հարց տալ
response = await engine.query("Ինչպե՞ս կարող եմ ավտոմատացնել իմ ամենօրյա խնդիրները")
print(response['answer'])
```

---

## 🧠 Նոր Ալգորիթմներ

### 1. Priority Experience Replay
```python
# Կարևոր փորձերը պահվում են առանձին buffer-ում
# և ավելի հաճախ են օգտագործվում ուսուցման ժամանակ
if abs(experience.reward) > 0.8:
    priority_buffer.append(experience)
```

### 2. Dynamic Learning Rate Adjustment
```python
# Learning rate-ը ավտոմատ ճշգրտվում է ըստ performance-ի
trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
if trend > 0.01:
    learning_rate *= 1.05  # Increase if improving
elif trend < -0.01:
    learning_rate *= 0.95  # Decrease if unstable
```

### 3. Semantic Knowledge Search
```python
# Օգտագործում է sentence embeddings նմանատիպ գիտելիքներ գտնելու համար
query_embedding = model.encode(query)
similarity = cosine_similarity(query_embedding, stored_embeddings)
```

### 4. Multi-Step Reasoning Chains
```python
# Կառուցում է տրամաբանական շղթաներ որոշումների համար
for depth in range(reasoning_depth):
    next_thought = generate_next_thought(current_thought)
    chain.append(next_thought)
conclusion = derive_conclusion(chain)
```

---

## ⚙️ Կոնֆիգուրացիա

Ստեղծեք `config.yaml` ֆայլ.

```yaml
system:
  name: "Nexus Core"
  debug_mode: false
  max_concurrent_tasks: 10

learning:
  auto_learn: true
  model_save_path: "./models"
  knowledge_ttl_days: 30

automation:
  safe_mode: true
  timeout_seconds: 300

storage:
  path: "./nexus_memory"
  backup_enabled: true
```

---

## 🔒 Անվտանգություն

- **Safe Mode**: Արգելափակում է վտանգավոր հրամանները
- **Allowed Directories**: Սահմանափակում է ֆայլային օպերացիաները
- **Command Filtering**: Արգելափակում է `rm -rf /`, `format` և այլ վտանգավոր հրամաններ

---

## 📊 Performance

| Բաղադրիչ | Նկարագրություն |
|-----------|----------------|
| Cognitive Processor | ~50ms response time |
| Task Executor | Async, up to 10 concurrent tasks |
| Adaptive Learner | Online learning, updates every 100 experiences |
| Knowledge Base | Semantic search with <100ms latency |
| Multi-Modal Analyzer | Supports text, image, audio, video |

---

## 🛠️ API (ապագա)

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/task")
async def submit_task(task: TaskSchema):
    engine = get_engine()
    task_id = await engine.submit_task(task.dict())
    return {"task_id": task_id}

@app.get("/status")
async def get_status():
    engine = get_engine()
    return engine.get_status()
```

---

## 📝 License

MIT License - ազատ օգտագործում և մոդիֆիկացիա

---

## 👨‍💻 Հեղինակ

Nexus AI Development Team

---

## 🌟 Ապագա Պլաններ

- [ ] Ինտեգրում Large Language Models-ի հետ
- [ ] Distributed computing support
- [ ] Real-time collaboration features
- [ ] Mobile application
- [ ] Plugin system ընդլայնման համար
- [ ] Visual programming interface

---

**Nexus Core** - Ձեր ամբողջական AI օգնականը համակարգչային խնդիրների համար! 🚀
