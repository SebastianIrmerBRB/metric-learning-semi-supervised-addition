import torch

from pytorch_metric_learning.utils import accuracy_calculator

# Your implementation copied from utils/__init__.py
def vectorized_get_relevance_mask(
    shape,
    gt_labels,
    ref_includes_query,
    label_counts,
):
    unique_labels, match_counts = label_counts

    query_labels = gt_labels.reshape(len(gt_labels), -1)
    unique_labels = unique_labels.reshape(len(unique_labels), -1)

    matches = (
        query_labels.unsqueeze(1) == unique_labels.unsqueeze(0)
    ).all(dim=2)

    relevant_count = (
        match_counts[matches.to(torch.uint8).argmax(dim=1)]
        - int(ref_includes_query)
    )

    count_per_query = torch.where(
        matches.any(dim=1),
        relevant_count,
        torch.zeros_like(relevant_count),
    )

    columns = torch.arange(shape[1], device=gt_labels.device)
    relevance_mask = columns.unsqueeze(0) < count_per_query.unsqueeze(1)

    return relevance_mask, count_per_query


def test_once(num_samples, num_classes, ref_includes_query, device):
    # Random labels
    labels = torch.randint(
        0,
        num_classes,
        (num_samples,),
        device=device,
    )

    # This is how PML represents label counts
    unique_labels, counts = torch.unique(
        labels,
        return_counts=True,
    )
    label_counts = (unique_labels, counts)

    # get_relevance_mask receives labels as (N, 1)
    gt_labels = labels[:, None]

    # Need enough columns to cover the maximum possible R
    max_count = int(counts.max())
    k = max_count - int(ref_includes_query)
    k = max(k, 1)

    shape = (num_samples, k)

    # Original PML implementation
    mask_pml, r_pml = accuracy_calculator.get_relevance_mask(
        shape,
        gt_labels,
        ref_includes_query,
        label_counts,
    )

    # Your vectorized implementation
    mask_vec, r_vec = vectorized_get_relevance_mask(
        shape,
        gt_labels,
        ref_includes_query,
        label_counts,
    )

    assert torch.equal(r_pml, r_vec), (
        f"R mismatch!\n"
        f"PML: {r_pml}\n"
        f"Vec: {r_vec}"
    )

    assert torch.equal(mask_pml, mask_vec), (
        f"Mask mismatch!\n"
        f"PML:\n{mask_pml}\n"
        f"Vec:\n{mask_vec}"
    )


device = "cuda" if torch.cuda.is_available() else "cpu"

for seed in range(1000):
    torch.manual_seed(seed)

    for ref_includes_query in [False, True]:
        test_once(
            num_samples=1000,
            num_classes=100,
            ref_includes_query=ref_includes_query,
            device=device,
        )

print("All tests passed: implementations are exactly equivalent.")