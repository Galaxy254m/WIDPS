"""Train + evaluate the classifier.  python train.py            (simulated data)
                                    python train.py --pcap lab.pcap --attack 20-35,60-80
Train and test come from DIFFERENT capture sessions (different seeds / files)
so the score isn't inflated by near-duplicate neighbouring windows."""
import argparse, joblib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
import Wids

ap = argparse.ArgumentParser()
ap.add_argument("--pcap"); ap.add_argument("--awid")
ap.add_argument("--attack", help="attack intervals in seconds, e.g. 20-35,60-80")
ap.add_argument("--model-out", default="wids_model.joblib")
a = ap.parse_args()

if a.pcap or a.awid:
    fr = Wids.pcap_to_frames(a.pcap) if a.pcap else Wids.awid3_csv_to_frames(a.awid)
    if a.attack:
        fr = Wids.label_by_intervals(fr, [tuple(map(float, s.split("-"))) for s in a.attack.split(",")])
    X = Wids.extract_features(fr)
    # time-ordered split: first 70% of the capture trains, last 30% tests
    X = X.sort_values("win_start"); cut = int(len(X) * 0.7)
    tr, te = X.iloc[:cut], X.iloc[cut:]
else:
    tr = Wids.extract_features(Wids.simulate(duration=600, seed=1, n_attacks=14))
    te = Wids.extract_features(Wids.simulate(duration=300, seed=2, n_attacks=8))

print(f"train windows={len(tr)} (attack={int(tr.label.sum())})  "
      f"test windows={len(te)} (attack={int(te.label.sum())})")
Xtr, ytr, Xte, yte = tr[Wids.FEATURES], tr.label, te[Wids.FEATURES], te.label

models = {"DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=0, class_weight="balanced"),
          "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, class_weight="balanced")}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:13s} test accuracy = {accuracy_score(yte, m.predict(Xte)):.4f}")

best = models["RandomForest"]
pred = best.predict(Xte)
print("\nRandomForest report:\n", classification_report(yte, pred, target_names=["normal", "malicious"], digits=3))
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(yte, pred))
print("\nDecision-tree rules (human readable):\n",
      export_text(models["DecisionTree"], feature_names=Wids.FEATURES))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["normal", "malicious"], cmap="Blues", ax=ax[0], colorbar=False)
ax[0].set_title("Random Forest - confusion matrix (test set)")
imp = sorted(zip(Wids.FEATURES, best.feature_importances_), key=lambda x: x[1])
ax[1].barh([i[0] for i in imp], [i[1] for i in imp], color="#2a6f97"); ax[1].set_title("Feature importance")
plt.tight_layout(); plt.savefig("results.png", dpi=200)
joblib.dump(best, a.model_out); print("saved", a.model_out, "and results.png")