#!/usr/bin/env python3
"""Serve Trial Atlas and expose selected TensorBoard scalars as compact JSON.

The browser cannot read TensorBoard event files directly. This local companion
service maps a trial's ``attr:log_dir`` to the mirrored ``tensorboard-files``
tree, extracts only the curves used by the dashboard, and serves the static
React export from the same origin.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

try:
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.util import tensor_util
except ImportError as exc:  # pragma: no cover - startup dependency check
    raise SystemExit(
        "TensorBoard is required. Run this service with the repository's .venv, "
        "or install dependencies with `python -m pip install tensorboard`."
    ) from exc


EVENT_PATTERN = "events.out.tfevents.*"
MAX_CACHE_ENTRIES = 64

VALIDATION_LOSS_TAGS = ("epoch/valid/loss", "valid/loss", "validation/loss")
VALIDATION_METRIC_TAGS = (
    "epoch/valid/mean_average_precision_at_r",
    "valid/mean_average_precision_at_r",
    "epoch/valid/precision_at_1",
    "valid/precision_at_1",
)
BATCH_LOSS_TAGS = ("train/batch_loss",)
EPOCH_LOSS_TAGS = ("epoch/train/loss", "train/epoch_loss")
TARGET_TAGS = set(
    (*VALIDATION_LOSS_TAGS, *VALIDATION_METRIC_TAGS, *BATCH_LOSS_TAGS, *EPOCH_LOSS_TAGS)
)


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def _clean_log_dir(value: str) -> str:
    cleaned = str(value).strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned.replace("\\", "/")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_trial_directory(root: Path, raw_log_dir: str) -> Path:
    """Map logs/<suffix> or tensorboard-files/<suffix> into the configured root."""

    cleaned = _clean_log_dir(raw_log_dir)
    if not cleaned:
        raise ValueError("The selected trial does not contain a log directory.")

    parts = [part for part in cleaned.split("/") if part not in {"", "."}]
    lower_parts = [part.lower() for part in parts]
    if "logs" in lower_parts:
        parts = parts[lower_parts.index("logs") + 1 :]
    elif "tensorboard-files" in lower_parts:
        parts = parts[lower_parts.index("tensorboard-files") + 1 :]
    elif Path(cleaned).is_absolute():
        raise ValueError("Absolute log paths must contain a logs or tensorboard-files segment.")

    if not parts or any(part == ".." for part in parts):
        raise ValueError("The trial log path is invalid.")

    candidate = root.joinpath(*parts).resolve()
    if not _is_relative_to(candidate, root):
        raise ValueError("The trial log path is outside the configured TensorBoard root.")
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"No mirrored TensorBoard directory exists for this trial: {candidate.relative_to(root)}"
        )
    return candidate


def _fold_id(trial_directory: Path, event_file: Path) -> str:
    relative_parts = event_file.relative_to(trial_directory).parts
    for part in relative_parts:
        if re.fullmatch(r"fold[_-]?\d+", part, flags=re.IGNORECASE):
            return part.lower().replace("-", "_")
    return "run"


def _scalar_value(summary_value) -> float | None:
    try:
        if summary_value.HasField("simple_value"):
            value = float(summary_value.simple_value)
        elif summary_value.HasField("tensor"):
            array = tensor_util.make_ndarray(summary_value.tensor).reshape(-1)
            if array.size != 1:
                return None
            value = float(array[0])
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _load_selected_scalars(event_files: Iterable[Path]) -> tuple[dict[str, list[ScalarPoint]], list[str]]:
    # Later wall times win when a resumed event file rewrites an existing step.
    by_tag: dict[str, dict[int, ScalarPoint]] = defaultdict(dict)
    warnings: list[str] = []

    for event_file in sorted(event_files, key=lambda path: (path.stat().st_mtime_ns, path.name)):
        try:
            for event in LegacyEventFileLoader(str(event_file)).Load():
                if not event.summary:
                    continue
                for summary_value in event.summary.value:
                    tag = summary_value.tag
                    if tag not in TARGET_TAGS:
                        continue
                    value = _scalar_value(summary_value)
                    if value is None:
                        continue
                    point = ScalarPoint(int(event.step), value, float(event.wall_time))
                    current = by_tag[tag].get(point.step)
                    if current is None or point.wall_time >= current.wall_time:
                        by_tag[tag][point.step] = point
        except Exception as exc:  # A live file may end in a partial record.
            warnings.append(f"Could not completely read {event_file.name}: {exc}")

    return {
        tag: sorted(points.values(), key=lambda point: (point.step, point.wall_time))
        for tag, points in by_tag.items()
    }, warnings


def _min_max_downsample(points: list[ScalarPoint], maximum: int) -> list[ScalarPoint]:
    """Keep the shape and spikes of a dense scalar series within a point budget."""

    if len(points) <= maximum or maximum < 4:
        return points

    interior = points[1:-1]
    bucket_count = max(1, (maximum - 2) // 2)
    result = [points[0]]
    for bucket in range(bucket_count):
        start = round(bucket * len(interior) / bucket_count)
        end = round((bucket + 1) * len(interior) / bucket_count)
        values = interior[start:end]
        if not values:
            continue
        low = min(values, key=lambda point: point.value)
        high = max(values, key=lambda point: point.value)
        result.extend(sorted({low, high}, key=lambda point: (point.step, point.wall_time)))
    result.append(points[-1])
    return result[:maximum]


def _pick_series(
    values: dict[str, list[ScalarPoint]], candidates: Iterable[str]
) -> tuple[str, list[ScalarPoint]] | None:
    for tag in candidates:
        points = values.get(tag)
        if points:
            return tag, points
    return None


def _series_payload(
    selected: tuple[str, list[ScalarPoint]] | None,
    label: str,
    maximum_points: int,
    epoch_axis: bool = False,
) -> dict | None:
    if selected is None:
        return None
    tag, original = selected
    points = original
    use_epoch_index = epoch_axis and not tag.startswith("epoch/")
    if use_epoch_index:
        points = [ScalarPoint(index, point.value, point.wall_time) for index, point in enumerate(points)]
    sampled = _min_max_downsample(points, maximum_points)
    return {
        "tag": tag,
        "label": label,
        "x_label": "Epoch" if epoch_axis else "Optimizer step",
        "point_count": len(original),
        "sampled_count": len(sampled),
        "points": [[point.step, point.value, point.wall_time] for point in sampled],
    }


class TrialDataCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple, dict] = {}

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            return self._values.get(key)

    def put(self, key: tuple, value: dict) -> None:
        with self._lock:
            if len(self._values) >= MAX_CACHE_ENTRIES:
                self._values.pop(next(iter(self._values)))
            self._values[key] = value


def build_trial_payload(root: Path, raw_log_dir: str, maximum_points: int, cache: TrialDataCache) -> dict:
    trial_directory = resolve_trial_directory(root, raw_log_dir)
    event_files = sorted(trial_directory.rglob(EVENT_PATTERN))
    if not event_files:
        raise FileNotFoundError("No events.out.tfevents.* files were found for this trial.")

    signatures = tuple(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in event_files
    )
    cache_key = (str(trial_directory), maximum_points, signatures)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    grouped: dict[str, list[Path]] = defaultdict(list)
    for event_file in event_files:
        grouped[_fold_id(trial_directory, event_file)].append(event_file)

    folds: list[dict] = []
    warnings: list[str] = []
    used_validation_fallback = False
    for fold_id in sorted(grouped):
        values, read_warnings = _load_selected_scalars(grouped[fold_id])
        warnings.extend(read_warnings)

        validation = _pick_series(values, VALIDATION_LOSS_TAGS)
        validation_label = "Validation loss"
        if validation is None:
            validation = _pick_series(values, VALIDATION_METRIC_TAGS)
            if validation is not None:
                used_validation_fallback = True
                validation_label = (
                    "Validation mAP@R" if "mean_average_precision" in validation[0] else "Validation precision@1"
                )

        folds.append(
            {
                "id": fold_id,
                "label": "Run" if fold_id == "run" else fold_id.replace("_", " ").title(),
                "event_file_count": len(grouped[fold_id]),
                "series": {
                    "validation": _series_payload(
                        validation,
                        validation_label,
                        maximum_points,
                        epoch_axis=True,
                    ),
                    "batch_loss": _series_payload(
                        _pick_series(values, BATCH_LOSS_TAGS),
                        "Batch loss",
                        maximum_points,
                    ),
                    "epoch_loss": _series_payload(
                        _pick_series(values, EPOCH_LOSS_TAGS),
                        "Epoch loss",
                        maximum_points,
                        epoch_axis=True,
                    ),
                },
            }
        )

    if used_validation_fallback:
        warnings.insert(
            0,
            "Validation loss was not logged for this trial; the validation retrieval metric is shown instead.",
        )

    payload = {
        "status": "ok",
        "log_dir": _clean_log_dir(raw_log_dir),
        "resolved_path": str(trial_directory.relative_to(root)).replace("\\", "/"),
        "event_file_count": len(event_files),
        "folds": folds,
        "warnings": warnings,
    }
    cache.put(cache_key, payload)
    return payload


class TrialAtlasHandler(SimpleHTTPRequestHandler):
    server_version = "TrialAtlas/1.0"

    def __init__(self, *args, directory: str | None = None, **kwargs) -> None:
        super().__init__(*args, directory=directory, **kwargs)

    @property
    def app_server(self):
        return self.server

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler contract
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "tensorboard_root": str(self.app_server.tensorboard_root),
                    "site_available": self.app_server.static_directory.is_dir(),
                },
            )
            return

        if parsed.path == "/api/tensorboard/trial":
            query = parse_qs(parsed.query)
            log_dir = query.get("log_dir", [""])[0]
            try:
                maximum_points = int(query.get("max_points", ["2000"])[0])
                maximum_points = min(5000, max(200, maximum_points))
                payload = build_trial_payload(
                    self.app_server.tensorboard_root,
                    log_dir,
                    maximum_points,
                    self.app_server.trial_cache,
                )
                self._send_json(HTTPStatus.OK, payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"status": "error", "message": f"TensorBoard data could not be read: {exc}"},
                )
            return

        if parsed.path.rstrip("/") == "/api/embedding-image":
            query = parse_qs(parsed.query)
            self._send_embedding_image(
                query.get("run", [""])[0],
                query.get("set", [""])[0],
                query.get("sample", [""])[0],
            )
            return

        super().do_GET()

    def _send_embedding_image(self, run: str, image_set: str, sample: str) -> None:
        """Serve the data-folder image behind one viewer point, from inside the repository only."""

        root = self.app_server.image_root.resolve()
        if not run or image_set not in IMAGE_SETS or not sample.isdigit():
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "message": "Expected run, set (train, validation or test) and a sample index"},
            )
            return
        try:
            index = self.app_server.image_indexes.get((root / run).resolve())
        except RuntimeError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": str(exc)})
            return
        paths = index.get(image_set) or []
        requested = paths[int(sample)] if int(sample) < len(paths) else None
        if not requested:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "This sample has no image file"})
            return
        resolved = (root / requested).resolve()
        content_type = IMAGE_CONTENT_TYPES.get(resolved.suffix.lower())
        if content_type is None or root not in resolved.parents:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "Not an image inside the repository"})
            return
        try:
            body = resolved.read_bytes()
        except OSError:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Image not found"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(body)


IMAGE_SETS = ("train", "validation", "test")


class ImageIndexCache:
    """Each run's image files per set, rebuilt from the data folder once per run.

    evaluate/embedding_image_index.py runs in the repository's venv, so this
    server never imports torch; the lists stay in memory only.
    """

    def __init__(self, repository: Path) -> None:
        self.repository = repository
        self._indexes: dict[Path, dict] = {}
        self._locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()

    def _python(self) -> str:
        venv = self.repository / "venv" / "bin" / "python"
        return str(venv) if venv.is_file() else sys.executable

    def get(self, run_dir: Path) -> dict:
        with self._guard:
            lock = self._locks[run_dir]
        # One rebuild per run even when a gallery requests many images at once.
        with lock:
            if run_dir not in self._indexes:
                completed = subprocess.run(
                    [self._python(), "-m", "evaluate.embedding_image_index", str(run_dir)],
                    cwd=self.repository,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    reason = (completed.stderr.strip().splitlines() or ["unknown error"])[-1]
                    raise RuntimeError(f"The run's datasets could not be rebuilt: {reason}")
                self._indexes[run_dir] = json.loads(completed.stdout)
            return self._indexes[run_dir]


IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


class TrialAtlasServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        handler,
        tensorboard_root: Path,
        static_directory: Path,
        image_root: Path | None = None,
    ) -> None:
        super().__init__(address, handler)
        self.tensorboard_root = tensorboard_root
        self.static_directory = static_directory
        # Dataset images are served from inside the repository only.
        self.image_root = image_root or Path(__file__).resolve().parent.parent
        self.image_indexes = ImageIndexCache(self.image_root)
        self.trial_cache = TrialDataCache()


def parse_args() -> argparse.Namespace:
    project_directory = Path(__file__).resolve().parent
    repository_directory = project_directory.parent
    parser = argparse.ArgumentParser(description="Serve Trial Atlas with TensorBoard scalar data.")
    parser.add_argument("--root", type=Path, default=repository_directory / "tensorboard-files")
    parser.add_argument("--static-dir", type=Path, default=project_directory / "out")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    static_directory = args.static_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"TensorBoard root does not exist: {root}")

    handler = lambda *handler_args, **handler_kwargs: TrialAtlasHandler(  # noqa: E731
        *handler_args,
        directory=str(static_directory),
        **handler_kwargs,
    )
    server = TrialAtlasServer((args.host, args.port), handler, root, static_directory)
    print(f"Trial Atlas: http://{args.host}:{args.port}/")
    print(f"TensorBoard root: {root}")
    if not static_directory.is_dir():
        print(f"Warning: static site not found at {static_directory}; run `npm run build` first.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
