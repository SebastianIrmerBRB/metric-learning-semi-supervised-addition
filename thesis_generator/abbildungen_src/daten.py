"""The published Cars196 queries and six neighbors, from the compact bundle."""
from functools import lru_cache
import io

from PIL import Image
import datensammlung as sammlung


@lru_cache(maxsize=1)
def cars_test():
    r = sammlung.read_csv("qualitative/cars_test.csv")
    return r.path.to_numpy(), r.label.to_numpy()


def neighbours(method, q, n):
    r = sammlung.read_csv("qualitative/retrieval_neighbors.csv")
    x = r[(r.method == method) & (r["query"] == q)].sort_values("rank")
    if len(x) < n:
        raise ValueError(f"Query {q}/{method}, {n} neighbors is not in the data bundle (included: 758, 5348; six neighbors).")
    return x["index"].to_numpy()[:n]


def image(path):
    paths, _ = cars_test()
    idx = list(paths).index(path)
    return Image.open(io.BytesIO(sammlung.asset_bytes(f"qualitative/images/{idx}.png"))).convert("RGB")
