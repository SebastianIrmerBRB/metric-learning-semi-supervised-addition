"""Check the installed runtime before downloading data or starting training."""

from __future__ import annotations

import argparse
from importlib import import_module, metadata
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also require a working visible NVIDIA GPU")
    args = parser.parse_args()
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 12):
        print("Use Python 3.12 or newer; the pinned environment was tested with Python 3.12.")
        return 1
    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "pytorch-metric-learning": "pytorch_metric_learning",
        "faiss-cpu": "faiss",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "Pillow": "PIL",
        "optuna": "optuna",
        "pacmap": "pacmap",
        "tensorboard": "tensorboard",
    }
    errors = []
    modules = {}
    for distribution, module in packages.items():
        try:
            modules[module] = import_module(module)
            print(f"OK {distribution} {metadata.version(distribution)}")
        except Exception as exc:
            errors.append(f"{distribution}: {exc}")
    if errors:
        print("\nEnvironment errors:")
        print("\n".join(errors))
        print("Install requirements.txt in a fresh virtual environment, then retry.")
        return 1
    torch, np, faiss = modules["torch"], modules["numpy"], modules["faiss"]
    try:
        vectors = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        index = faiss.IndexFlatL2(2)
        index.add(vectors)
        _, neighbors = index.search(vectors[:1], 1)
        if neighbors[0, 0] != 0:
            raise RuntimeError("FAISS did not return the expected nearest neighbor")
        if args.cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Use a fresh environment with requirements-cuda.txt "
                    "and a compatible NVIDIA driver, or run the CPU quickstart."
                )
            value = torch.ones(2, device="cuda", requires_grad=True)
            value.square().sum().backward()
            torch.cuda.synchronize()
            print(f"OK CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
        else:
            value = torch.ones(2, requires_grad=True)
            value.square().sum().backward()
            print("OK CPU tensor operations and gradients")
    except Exception as exc:
        print(f"Runtime check failed: {exc}")
        return 1
    print("Environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
