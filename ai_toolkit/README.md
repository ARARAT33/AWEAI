# AI Toolkit - Հզոր գործիք AI մշակողների համար

**Full-featured AI development toolkit with comprehensive modules**

## 📋 Features

### 🔹 Data Collection & Preprocessing
- Collect data from APIs, files (CSV, JSON, TXT), and web scraping
- Store collected data in SQLite database
- Export data to JSON format
- Data preprocessing: cleaning, normalization, deduplication
- Train/validation/test dataset splitting

### 🔹 Model Training & Evaluation
- Support for scikit-learn models (Logistic Regression, Random Forest, SVM, MLP)
- TensorFlow/Keras and PyTorch support (extensible)
- Hyperparameter tuning with Grid Search and Random Search
- Comprehensive model evaluation metrics
- Model saving and loading

### 🔹 Device Control & IoT
- USB and Serial device scanning and control
- GPIO pin control for Raspberry Pi
- Sensor data reading
- Automation rules engine
- Device history logging

### 🔹 Module Management & Plugins
- Dynamic module discovery and loading
- Plugin system with priority-based execution
- Hook system for extensibility
- Module registry and export

### 🔹 Analytics & Visualization
- Descriptive statistics
- Correlation analysis
- Outlier detection
- Chart data generation (bar, line, pie, scatter, heatmap)
- HTML report generation

### 🔹 Task Scheduling & Progress Tracking
- Multi-threaded task execution
- Scheduled and recurring tasks
- Progress tracking with ETA
- Configuration management
- Comprehensive logging

## 🚀 Installation

```bash
# Clone or copy the ai_toolkit directory to your project
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH
```

## 📖 Usage Examples

### Data Collection
```python
from ai_toolkit import DataCollector, DataPreprocessor

# Initialize collector
collector = DataCollector(storage_path="./my_data")

# Collect from file
collector.collect_from_file("data.csv", file_type="csv")

# Get statistics
stats = collector.get_statistics()
print(f"Total records: {stats['total_records']}")

# Export data
collector.export_to_json("export_data.json")

# Preprocess data
preprocessor = DataPreprocessor()
cleaned = preprocessor.clean_text("  Hello   World!  ")
```

### Model Training
```python
from ai_toolkit import ModelTrainer, HyperparameterTuner

# Initialize trainer
trainer = ModelTrainer(model_type="sklearn")

# Create model
trainer.create_model("random_forest_classifier", n_estimators=100)

# Train model
metrics = trainer.train(X_train, y_train, X_val, y_val)
print(f"Train score: {metrics['train_score']}")

# Evaluate
eval_results = trainer.evaluate(X_test, y_test)

# Save model
trainer.save_model("my_model.pkl")
```

### Device Control
```python
from ai_toolkit import DeviceController

# Initialize controller
controller = DeviceController()

# Scan devices
devices = controller.scan_devices(connection_type="serial")

# Connect and send command
controller.connect_device(device_id)
result = controller.send_command(device_id, "AT+STATUS")

# Read sensor data
sensor_data = controller.read_sensor_data(device_id)
```

### Module Management
```python
from ai_toolkit import ModuleManager

# Initialize manager
manager = ModuleManager(modules_path="./modules")

# Discover modules
discovered = manager.discover_modules()

# Load module
module = manager.load_module("data_collection")

# Execute function
result = manager.execute_module_function(
    "data_collection", 
    "some_function",
    arg1, arg2
)
```

### Logging & Configuration
```python
from ai_toolkit import Logger, ConfigManager

# Logger
logger = Logger(name="my_app", log_file="app.log")
logger.info("Application started")
logger.error("Something went wrong")

# Configuration
config = ConfigManager("config.json")
config.set("api_key", "secret", section="auth")
config.set("timeout", 30)

value = config.get("api_key", section="auth")
config.save_config()
```

### Progress Tracking
```python
from ai_toolkit import ProgressTracker

tracker = ProgressTracker(total=100, description="Processing")
tracker.start()

for i in range(100):
    # Do work
    tracker.update(1)
    tracker.print_progress()

status = tracker.get_status()
print(f"Completed: {status['percentage']}%")
```

### Analytics & Visualization
```python
from ai_toolkit import DataAnalyzer, VisualizationTools, ReportGenerator

# Analyze data
analyzer = DataAnalyzer()
stats = analyzer.descriptive_statistics([1, 2, 3, 4, 5])
outliers = analyzer.detect_outliers([1, 2, 3, 100])

# Create visualizations
viz = VisualizationTools()
viz.create_bar_chart_data(['A', 'B', 'C'], [10, 20, 30])
viz.create_line_chart_data([1, 2, 3], [4, 5, 6])
viz.export_all_charts()

# Generate reports
reporter = ReportGenerator()
report = reporter.create_analysis_report(
    title="Analysis Report",
    data_summary=stats,
    findings=["Finding 1", "Finding 2"],
    recommendations=["Recommendation 1"]
)
reporter.export_report_markdown(0)
```

## 📁 Project Structure

```
ai_toolkit/
├── __init__.py              # Main package initialization
├── modules/
│   ├── __init__.py
│   ├── data_collection.py   # Data collection & preprocessing
│   ├── model_training.py    # Model training & evaluation
│   ├── device_control.py    # Device & IoT control
│   └── modules.py           # Module & plugin management
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Logger, Config, Tasks, Progress
│   └── analytics.py         # Analytics & visualization
├── data/                    # Data storage
└── output/                  # Output files
    ├── models/
    ├── visualizations/
    └── reports/
```

## 🎯 Key Classes

| Class | Description |
|-------|-------------|
| `DataCollector` | Collect data from multiple sources |
| `DataPreprocessor` | Clean and preprocess data |
| `ModelTrainer` | Train ML models |
| `HyperparameterTuner` | Optimize hyperparameters |
| `DeviceController` | Control hardware devices |
| `ModuleManager` | Manage dynamic modules |
| `PluginSystem` | Plugin architecture |
| `Logger` | Advanced logging |
| `ConfigManager` | Configuration management |
| `TaskScheduler` | Schedule and run tasks |
| `ProgressTracker` | Track progress with ETA |
| `DataAnalyzer` | Statistical analysis |
| `VisualizationTools` | Create chart data |
| `ReportGenerator` | Generate reports |

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to add new modules, improve existing ones, or report issues.

---

**AI Toolkit** - Everything you need for AI development in one powerful package! 🚀
