"""Demo: train a small MLP on XOR from scratch, evaluate, export."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aweai.hardware import detect
from aweai.models.registry import recommended_for_task
from aweai.train import train
from aweai.management import load_model, export_model
from aweai.eval import classification_report


def main() -> None:
    print("Hardware:", detect().to_dict())
    print("Recommended:", recommended_for_task("classification"))
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    print("\nTraining MLP on XOR…")
    res = train("mlp", "xor_demo", X=X, y=y, params={"epochs": 60, "hidden": [4, 2], "lr": 0.3})
    print(f"Trained {res.name} v{res.version} — loss {res.loss:.4f} in {res.duration_s:.2f}s")
    model, meta = load_model("xor_demo")
    print("Predictions:", model.predict(X).tolist())
    print("Report:", classification_report(y, model.predict(X)))
    out = export_model("xor_demo", fmt="json")
    print("Exported:", out)


if __name__ == "__main__":
    main()
