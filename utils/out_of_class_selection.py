"""Deterministic, class-balanced subsetting of an out-of-class unlabeled pool.

Class-mismatch experiments vary how much out-of-class data reaches the
unlabeled pool. Both Semi-Aves and Semi-iNat draw that subset the same way, so
the rule lives here instead of being reimplemented per dataset.
"""

import hashlib


def select_out_of_class_records(
    records,
    fraction,
    seed=0,
    *,
    selection_version,
    dataset_label,
):
    """Keep a class-balanced ``fraction`` of an out-of-class record list.

    The kept images are spread evenly over the out-of-class species instead of
    concentrated in whichever ones happen to be large. Selection is a pure
    function of the image path, the class, and ``seed``, so a fraction always
    yields the same images, and a larger fraction reuses a smaller one's images
    up to the per-class rounding.

    ``records`` are ``(relative_path, label, source)`` triples.
    """

    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{dataset_label} out-of-class fraction must be in [0, 1]: {fraction}"
        )
    records = list(records)
    if not records or fraction == 0.0:
        return []
    if fraction == 1.0:
        return records

    seed = int(seed)
    positions_by_class = {}
    for position, (relative_path, label, _) in enumerate(records):
        rank_key = hashlib.sha256(
            f"{selection_version}:{seed}:"
            f"{relative_path.as_posix()}".encode("utf-8")
        ).digest()
        positions_by_class.setdefault(int(label), []).append(
            (rank_key, position)
        )

    # Largest-remainder apportioning keeps every class's share proportional
    # while the kept total still matches the requested fraction exactly.
    target_total = int(round(len(records) * fraction))
    quotas = {}
    remainders = []
    for label, entries in positions_by_class.items():
        exact_quota = len(entries) * fraction
        quotas[label] = int(exact_quota)
        remainders.append((exact_quota - quotas[label], label))
    leftover = target_total - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, label in remainders[:leftover]:
        quotas[label] += 1

    kept_positions = []
    for label, entries in positions_by_class.items():
        entries.sort()
        kept_positions.extend(
            position for _, position in entries[: quotas[label]]
        )
    # Restore source order so manifests stay easy to compare across runs.
    kept_positions.sort()
    return [records[position] for position in kept_positions]
