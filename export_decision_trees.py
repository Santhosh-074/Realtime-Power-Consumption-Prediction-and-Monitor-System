from pathlib import Path
import sqlite3
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "instance" / "rtpcpms.db"
OUTPUT_DIR = ROOT / "instance" / "models" / "decision_trees"
TREE_COUNT = 100

# Select the most recently trained Random Forest model from this app's database.
with sqlite3.connect(DB_PATH) as connection:
    row = connection.execute(
        "SELECT path FROM models WHERE algorithm = ? ORDER BY trained_at DESC, id DESC LIMIT 1",
        ("random_forest",),
    ).fetchone()
if row is None:
    raise SystemExit("No Random Forest model is registered. Train one in the app first.")

model_path = Path(row[0])
if not model_path.is_absolute():
    model_path = ROOT / model_path
if not model_path.is_file():
    raise SystemExit(f"Saved model file was not found: {model_path}")

bundle = joblib.load(model_path)
forest = bundle["pipeline"].named_steps["model"]
if not isinstance(forest, RandomForestRegressor):
    raise SystemExit("The selected model file is not a Random Forest.")
if len(forest.estimators_) < TREE_COUNT:
    raise SystemExit(f"This forest has only {len(forest.estimators_)} trees; {TREE_COUNT} are required.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
feature_names = bundle.get("columns")
for index, tree in enumerate(forest.estimators_[:TREE_COUNT], start=1):
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True,
              max_depth=3, fontsize=7, ax=ax)
    ax.set_title(f"Random Forest - Decision Tree {index}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"tree_{index:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tree {index}/{TREE_COUNT}")

print(f"Finished. Model: {model_path}")
print(f"Tree images: {OUTPUT_DIR}")
