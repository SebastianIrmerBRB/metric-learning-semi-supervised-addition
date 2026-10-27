#!/usr/bin/env python3
"""Download + resize iNaturalist 2018 over many parallel connections.

Why this exists
---------------
dipa's uplink loses ~0.2% of packets on non-CDN paths. One CUBIC flow collapses
to ~0.08 MB/s (434 h for the archive); 64 flows reach several MB/s because each
absorbs its own losses independently. BBR would fix it properly but this kernel
only offers reno/cubic.

The archive is 128 GB and dipa has ~100 GB free, so it can never be stored. It
is also a single gzip stream, so ranged chunks cannot be decoded out of order.
This module therefore fetches many chunks in parallel, reassembles them strictly
in order behind a bounded buffer, and hands that to the *existing*
INaturalist2018 pipeline -- which streams the tar, resizes to 224x224, and keeps
only the ~8.3 GB of resized JPEGs.

Nothing in utils/local_datasets.py is modified: only _open_url is swapped, so
the progress bars, image-count validation and completion marker all still apply.

Usage
-----
    tmux new -s inat          # survive an ssh drop
    python fetch_inat2018_parallel.py --root data/inaturalist2018 --streams 64

Interrupting is safe. Already-resized images are detected and skipped on a
rerun, but note the archive itself must be re-streamed from the start (a gzip
stream has no seek points), so a restart re-downloads. Prefer letting it run.
"""

import argparse
import http.client
import os
import ssl
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.local_datasets import INaturalist2018  # noqa: E402

MB = 1024.0 * 1024.0


class ParallelRangeReader:
    """Sequential file-like view over many parallel ranged GETs.

    Exposes exactly what _stream_and_resize_images needs: a context manager with
    ``.headers`` carrying Content-Length and a blocking ``.read(size)`` that
    yields the object's bytes in order.
    """

    def __init__(self, url, streams=64, chunk=4 << 20, buffer_bytes=3 << 30,
                 timeout=40, max_attempts=8, report_every=30.0):
        if not url.startswith("https://"):
            raise ValueError(f"expected an https url, got {url!r}")
        remainder = url[len("https://"):]
        self.host, _, path = remainder.partition("/")
        self.path = "/" + path
        self.streams = streams
        self.chunk = chunk
        self.timeout = timeout
        self.max_attempts = max_attempts

        self.total = self._head()
        self.chunk_count = (self.total + chunk - 1) // chunk
        # How far workers may run ahead of the reader, in chunks. Never below
        # `streams`, or workers would idle; peak RAM is ahead * chunk.
        self.ahead = max(streams, int(buffer_bytes // chunk))
        self.headers = {"Content-Length": str(self.total)}

        self._lock = threading.Condition()
        self._ready = {}          # index -> bytes, waiting to be consumed
        self._next_issue = 0      # next index handed to a worker
        self._next_serve = 0      # next index the reader needs
        self._retry = []          # indices whose fetch failed, to reissue
        self._pending = bytearray()
        self._served = 0
        self._error = None
        self._closed = False
        self._threads = []
        self._fetched = 0
        self._chunks_done = 0
        self._started = time.monotonic()
        self._report_every = report_every
        self._stop = threading.Event()

    # ---------------------------------------------------------------- plumbing
    def _connect(self):
        conn = http.client.HTTPSConnection(
            self.host, timeout=self.timeout, context=ssl.create_default_context()
        )
        conn.connect()
        return conn

    def _head(self):
        conn = self._connect()
        try:
            conn.request("HEAD", self.path,
                         headers={"User-Agent": "inat-parallel/1.0"})
            response = conn.getresponse()
            if response.status != 200:
                raise OSError(f"HEAD {self.path} -> HTTP {response.status}")
            length = response.getheader("Content-Length")
            if not length:
                raise OSError("origin did not report Content-Length")
            response.read()
            return int(length)
        finally:
            conn.close()

    def _fetch(self, conn, index):
        """Fetch one chunk. Returns (conn, data); conn is None if unusable."""
        start = index * self.chunk
        end = min(start + self.chunk, self.total) - 1   # clamp: avoids HTTP 416
        try:
            if conn is None:
                conn = self._connect()
            conn.putrequest("GET", self.path, skip_host=True,
                            skip_accept_encoding=True)
            conn.putheader("Host", self.host)
            conn.putheader("Range", f"bytes={start}-{end}")
            conn.putheader("User-Agent", "inat-parallel/1.0")
            conn.endheaders()
            response = conn.getresponse()
            if response.status != 206:
                detail = response.read(300)
                raise OSError(f"HTTP {response.status} for chunk {index}: "
                              f"{detail[:200]!r}")
            want = end - start + 1
            parts = []
            got = 0
            while got < want:
                block = response.read(min(1 << 20, want - got))
                if not block:
                    raise OSError(f"chunk {index}: short read {got}/{want}")
                parts.append(block)
                got += len(block)
            return conn, b"".join(parts)
        except Exception:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise

    def _claim(self):
        """Pick the next index to fetch, respecting the look-ahead window."""
        with self._lock:
            while not self._closed and self._error is None:
                if self._retry:
                    return self._retry.pop(0)
                if self._next_issue >= self.chunk_count:
                    return None
                if self._next_issue < self._next_serve + self.ahead:
                    index = self._next_issue
                    self._next_issue += 1
                    return index
                self._lock.wait(0.5)   # buffer full: let the reader drain it
            return None

    def _worker(self):
        conn = None
        while True:
            index = self._claim()
            if index is None:
                break
            for attempt in range(self.max_attempts):
                try:
                    conn, data = self._fetch(conn, index)
                    with self._lock:
                        self._ready[index] = data
                        self._fetched += len(data)
                        self._chunks_done += 1
                        self._lock.notify_all()
                    break
                except Exception as exc:
                    conn = None
                    if attempt == self.max_attempts - 1:
                        with self._lock:
                            self._error = exc
                            self._lock.notify_all()
                        return
                    time.sleep(min(2.0 ** attempt, 20.0))
            with self._lock:
                if self._closed or self._error is not None:
                    break
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _monitor(self):
        """Report transport progress; the sequential bar lags behind by design."""
        try:
            from tqdm import tqdm
            emit = tqdm.write
        except ImportError:
            emit = print
        while not self._stop.wait(self._report_every):
            with self._lock:
                fetched, chunks = self._fetched, self._chunks_done
                serving, buffered = self._next_serve, len(self._ready)
            elapsed = time.monotonic() - self._started
            rate = fetched / elapsed / MB if elapsed > 0 else 0.0
            left = self.total - serving * self.chunk
            eta = left / (rate * MB) / 3600 if rate > 0 else float("inf")
            emit(f"  [transport] {chunks}/{self.chunk_count} chunks  "
                 f"{fetched / 1e9:.2f} GB  {rate:.2f} MB/s avg  "
                 f"serving #{serving}, {buffered} buffered  eta {eta:.1f} h")

    def start(self):
        for _ in range(self.streams):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)
        if self._report_every:
            threading.Thread(target=self._monitor, daemon=True).start()
        return self

    # -------------------------------------------------------------- public api
    def read(self, size=-1):
        if size is None or size < 0:
            size = 1 << 24
        while len(self._pending) < size:
            with self._lock:
                if self._error is not None:
                    raise self._error
                chunk = self._ready.pop(self._next_serve, None)
                if chunk is None:
                    if self._next_serve >= self.chunk_count:
                        break            # genuine end of object
                    self._lock.wait(1.0)
                    continue
                self._next_serve += 1
                self._lock.notify_all()  # a slot freed up for the workers
            self._pending.extend(chunk)
        out = bytes(self._pending[:size])
        del self._pending[:size]
        self._served += len(out)
        return out

    def rate(self):
        elapsed = time.monotonic() - self._started
        return self._fetched / elapsed / MB if elapsed > 0 else 0.0

    def close(self):
        self._stop.set()
        with self._lock:
            self._closed = True
            self._lock.notify_all()
        for thread in self._threads:
            thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


# Set from the command line, read by the _open_url override below.
CONFIG = {"streams": 64, "chunk": 4 << 20, "buffer": 3 << 30}
ACTIVE = {}


class ParallelINaturalist2018(INaturalist2018):
    """Identical to the stock loader, but the big archive arrives in parallel."""

    @classmethod
    def _open_url(cls, url):
        if url == cls.IMG_DOWNLOAD_URL:
            reader = ParallelRangeReader(
                url,
                streams=CONFIG["streams"],
                chunk=CONFIG["chunk"],
                buffer_bytes=CONFIG["buffer"],
            ).start()
            ACTIVE["reader"] = reader
            print(f"  parallel transport: {CONFIG['streams']} streams, "
                  f"{CONFIG['chunk'] >> 20} MB chunks, "
                  f"{reader.chunk_count} chunks, "
                  f"{reader.ahead * CONFIG['chunk'] / 1e9:.1f} GB look-ahead",
                  flush=True)
            return reader
        return super()._open_url(url)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="dataset root (resized images land here)")
    parser.add_argument("--streams", type=int, default=64,
                        help="parallel connections (default 64)")
    parser.add_argument("--chunk-mb", type=int, default=4,
                        help="MB per ranged request (default 4). Each stream "
                             "runs at only ~0.07 MB/s here, and the reader "
                             "blocks until the *next* chunk completes, so big "
                             "chunks mean minutes of startup latency.")
    parser.add_argument("--buffer-gb", type=float, default=3.0,
                        help="in-order reassembly buffer (default 3)")
    args = parser.parse_args()

    CONFIG["streams"] = args.streams
    CONFIG["chunk"] = args.chunk_mb << 20
    CONFIG["buffer"] = int(args.buffer_gb * (1 << 30))

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    print(f"root      : {root}")
    print(f"free space: {free / 1e9:.1f} GB "
          f"(need ~9 GB; the 128 GB archive is never stored)")
    if free < 12e9:
        print("\nNot enough room for the resized set. Free up space first.")
        return 1

    if ParallelINaturalist2018.is_ready(root):
        print("Already complete -- nothing to do.")
        return 0

    began = time.monotonic()
    try:
        ParallelINaturalist2018.download_224(root)
    except KeyboardInterrupt:
        print("\ninterrupted; resized images already written are kept.")
        return 130
    finally:
        reader = ACTIVE.get("reader")
        if reader is not None:
            print(f"\ntransport average: {reader.rate():.2f} MB/s")
            reader.close()

    elapsed = time.monotonic() - began
    print(f"done in {elapsed / 3600:.2f} h")
    print(f"ready: {ParallelINaturalist2018.is_ready(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
