import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, TensorDataset


num_classes = 45
samples_per_class = 10000
feature_dim = 2
batch_size = 16
m = 8
length_before_new_iter = 4000
batches_to_show = 5
seed = 0
ProxyAnchorLoss

torch.manual_seed(seed)

labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
features = torch.randn(len(labels), feature_dim)
dataset = TensorDataset(features, labels)

sampler = MPerClassSampler(
    labels.tolist(),
    m=m,
    batch_size=batch_size,
    length_before_new_iter=length_before_new_iter,
)
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

print(f"dataset={len(dataset)} classes={num_classes} samples_per_class={samples_per_class}")
print(f"batch_size={batch_size} m={m} classes_per_batch={batch_size // m}")
print(f"sampler_len={len(sampler)} loader_batches={len(loader)}")

for batch_idx, (_, batch_labels) in enumerate(loader):
    counts = dict(zip(*torch.unique(batch_labels, return_counts=True)))
    counts = {int(label): int(count) for label, count in counts.items()}
    print(f"batch {batch_idx}: labels={batch_labels.tolist()} counts={counts}")

    if batch_idx + 1 >= batches_to_show:
        break
