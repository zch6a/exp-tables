"""examples/results.csv: made-up results for a made-up paper (four methods, two datasets, five seeds each), drawn
from fixed normal distributions with a fixed seed, so the file is the same every time. Not real results.

    uv run python scripts/make_example.py
"""

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(2026)
# (method, dataset) -> (accuracy mean, its spread, calibration error mean, its spread)
TRUTH = {
    ("TinyNet (ours)", "CIFAR-10"): (94.1, 0.15, 2.1, 0.20),
    ("ResNet-18", "CIFAR-10"): (93.2, 0.20, 3.4, 0.25),
    ("ViT-S", "CIFAR-10"): (93.6, 0.25, 2.9, 0.30),
    ("TinyNet w/o attention", "CIFAR-10"): (93.3, 0.20, 2.6, 0.20),
    ("TinyNet (ours)", "CIFAR-100"): (76.0, 0.45, 5.2, 0.40),
    ("ResNet-18", "CIFAR-100"): (73.8, 0.40, 7.9, 0.50),
    ("ViT-S", "CIFAR-100"): (75.7, 0.50, 6.1, 0.45),  # within the noise of ours on accuracy
    ("TinyNet w/o attention", "CIFAR-100"): (74.9, 0.40, 5.9, 0.35),
}
lines = ["method,dataset,seed,accuracy,ece"]
for (method, dataset), (acc, acc_sd, ece, ece_sd) in TRUTH.items():
    for seed in range(5):
        lines.append(f"{method},{dataset},{seed},{rng.gauss(acc, acc_sd):.2f},{max(0.1, rng.gauss(ece, ece_sd)):.2f}")
out = ROOT / "examples" / "results.csv"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}")
