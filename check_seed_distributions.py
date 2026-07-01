from pathlib import Path
import numpy as np

path = Path("data/StanfordOnlineProducts/Stanford_Online_Products/Ebay_train.txt")

rows = []
class_to_super = {}
super_to_name = {}

with path.open(encoding="utf-8") as f:
    next(f)
    for line in f:
        _, class_id, super_class_id, _ = line.split()[:4]
        class_id = int(class_id)
        super_class_id = int(super_class_id)
        super_class_name = Path(line.split()[3]).parts[0].replace("_final", "")
        rows.append(class_id)
        class_to_super[class_id] = super_class_id
        super_to_name[super_class_id] = super_class_name

labels = np.array(rows, dtype=np.int64)
rng = np.random.default_rng(20)

positions_by_label = {}
for label in np.unique(labels):
    class_positions = np.flatnonzero(labels == label)
    positions_by_label[int(label)] = rng.permutation(class_positions)

unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
num_selected = max(1, int(np.floor(len(unique_labels) * 0.01)))

selected = set(
    int(label)
    for label in rng.choice(unique_labels, size=num_selected, replace=False)
)

all_supers = set(class_to_super.values())
selected_supers = {class_to_super[label] for label in selected}
total_class_counts_by_super = {super_id: 0 for super_id in sorted(all_supers)}
selected_class_counts_by_super = {super_id: 0 for super_id in sorted(all_supers)}

for class_id, super_id in class_to_super.items():
    total_class_counts_by_super[super_id] += 1
    if class_id in selected:
        selected_class_counts_by_super[super_id] += 1

print("classes per superclass:")
for super_id in sorted(all_supers):
    print(
        f"  {super_id:2d} {super_to_name.get(super_id, '')}: "
        f"{selected_class_counts_by_super[super_id]} selected / "
        f"{total_class_counts_by_super[super_id]} total"
    )

print("selected classes:", num_selected)
print("covered superclasses:", sorted(selected_supers))
print("missing superclasses:", sorted(all_supers - selected_supers))
