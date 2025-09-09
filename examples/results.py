import json
import matplotlib.pyplot as plt

# ---------- 1. Load JSON ----------
with open("results.json") as f:
    results = json.load(f)

# ---------- 2. Prepare Figure ----------
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)

if n_models == 1:
    axes = [axes]  # make iterable if only one model

# ---------- 3. Plot per model ----------
for ax, model_entry in zip(axes, results):
    model_name = model_entry["model"]

    # Build labels & values for all metrics that have a mean
    labels, values = [], []
    for metric, val in model_entry.items():
        if metric == "model":
            continue
        mean = val.get("mean")
        if mean is not None:
            labels.append(metric)
            values.append(mean)

    # Bar plot
    x = range(len(labels))
    ax.bar(x, values, color=["#4caf50" if "digital" in l else
                             "#f44336" if "analog" in l else "#2196f3" for l in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title(model_name)
    ax.set_ylabel("Accuracy (%)")

plt.tight_layout()
plt.savefig("results.png", dpi=300)
plt.show()
