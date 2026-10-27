#!/usr/bin/env python3
"""Report CPU and memory usage on a shared Linux server.

Reads ``/proc`` directly, so it needs no third-party packages and runs on any
Linux box with Python 3.6+ -- including over a bare SSH session.

Three things this shows that a glance at ``top`` does not:

* **Load average per core.** CPU% answers "how busy are the cores"; load answers
  "how long is the queue". A machine at 40% CPU with a load of 3x its core count
  is thrashing, and that is usually what other users notice.
* **A per-user breakdown.** Your share of the machine, next to everyone else's.
* **Thread counts against the core budget.** PyTorch defaults to one compute
  thread per core *per process*, so N DataLoader workers or N concurrent trials
  can oversubscribe the machine by a factor of N while each process still looks
  reasonable on its own.

Only python processes are listed, since those are the training jobs; kernel
threads, editors and shells would otherwise fill the table. The machine-wide
CPU, memory and load lines still account for everything on the box, so nothing
is hidden -- pass ``--all`` to widen the listings too.

Usage::

    python3 scripts/server_usage.py                  # one 2-second sample
    python3 scripts/server_usage.py --watch          # refresh until Ctrl-C
    python3 scripts/server_usage.py --me             # only your processes
    python3 scripts/server_usage.py --all            # everything, not just python
    python3 scripts/server_usage.py --log usage.csv --watch --interval 60

The ``--log`` form is the one to use when the complaint is about something you
are not around to see: leave it running under ``nohup`` or ``tmux`` and read the
CSV afterwards.
"""

from __future__ import annotations

import argparse
import csv
import os
import pwd
import re
import sys
import time
from collections import defaultdict

# "python", "python3", "python3.12" -- but not "pythonish-tool".
PYTHON_NAME = re.compile(r"^python[\d.]*$")

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
GIB = 1024 ** 3

# Environment variables that cap per-process compute threads. Unset means the
# library picks "one per core", which is what oversubscribes a shared machine.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class Process:
    """A single snapshot of one process, as read from /proc/<pid>."""

    __slots__ = ("pid", "user", "comm", "cmdline", "cpu_ticks", "threads", "rss")

    def __init__(self, pid, user, comm, cmdline, cpu_ticks, threads, rss):
        self.pid = pid
        self.user = user
        self.comm = comm
        self.cmdline = cmdline
        self.cpu_ticks = cpu_ticks
        self.threads = threads
        self.rss = rss


def read_cpu_totals():
    """Aggregate jiffies from /proc/stat: (total, idle, per-field mapping)."""

    with open("/proc/stat") as handle:
        fields = [int(value) for value in handle.readline().split()[1:]]
    names = (
        "user", "nice", "system", "idle", "iowait",
        "irq", "softirq", "steal", "guest", "guest_nice",
    )
    breakdown = dict(zip(names, fields))
    # iowait is not the CPU doing work, but it is also not the machine being
    # healthy -- it is counted as idle here and reported separately below.
    idle = breakdown.get("idle", 0) + breakdown.get("iowait", 0)
    return sum(fields), idle, breakdown


def read_meminfo():
    """/proc/meminfo as a dict of bytes."""

    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                info[key] = int(parts[0]) * 1024
    return info


def read_cgroup_memory_limit():
    """Container/slurm memory ceiling, if one applies. None when unlimited."""

    candidates = (
        "/sys/fs/cgroup/memory.max",                    # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            limit = int(raw)
        except ValueError:
            continue
        # v1 reports a sentinel near 2**63 when unconstrained.
        if limit < (1 << 62):
            return limit
    return None


def read_process(pid):
    """Snapshot one pid, or None if it vanished or is unreadable."""

    try:
        with open(f"/proc/{pid}/stat") as handle:
            raw = handle.read()
        # comm can contain spaces and parentheses, so split on the *last* ')'.
        close = raw.rfind(")")
        comm = raw[raw.find("(") + 1:close]
        fields = raw[close + 2:].split()
        # /proc/<pid>/stat is 1-indexed and fields[0] is field 3, so field N
        # lives at fields[N - 3]: utime=14, stime=15, num_threads=20, rss=24.
        cpu_ticks = int(fields[11]) + int(fields[12])
        threads = int(fields[17])
        rss = int(fields[21]) * PAGE_SIZE

        uid = os.stat(f"/proc/{pid}").st_uid
        try:
            user = pwd.getpwuid(uid).pw_name
        except KeyError:
            user = str(uid)

        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmdline = ""
    except (OSError, ValueError, IndexError):
        return None
    return Process(pid, user, comm, cmdline, cpu_ticks, threads, rss)


def snapshot():
    """All readable processes, keyed by pid."""

    processes = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        process = read_process(int(entry))
        if process is not None:
            processes[process.pid] = process
    return processes


def is_python(process):
    """True for interpreter processes, including forked DataLoader workers.

    Checks both the executable name and argv[0]: a venv launcher shows up as
    ``/path/venv/bin/python``, while a shebang script keeps ``python3`` as its
    comm. Kernel threads have an empty cmdline and never match.
    """

    if PYTHON_NAME.match(process.comm):
        return True
    if not process.cmdline:
        return False
    return bool(PYTHON_NAME.match(os.path.basename(process.cmdline.split()[0])))


def thread_limits(pid):
    """Thread-cap env vars for a process we own. Empty when unreadable."""

    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            raw = handle.read()
    except OSError:
        return {}
    found = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.decode(errors="replace").partition("=")
        if sep and key in THREAD_VARS:
            found[key] = value
    return found


def measure(interval):
    """Sample twice `interval` apart and return usage derived from the delta.

    Per-process CPU% must come from a delta: the counters in /proc are
    cumulative, so reading them once gives a lifetime average, which is exactly
    the number that makes a busy machine look idle.
    """

    total_before, idle_before, _ = read_cpu_totals()
    before = snapshot()

    time.sleep(interval)

    total_after, idle_after, breakdown = read_cpu_totals()
    after = snapshot()

    total_delta = max(total_after - total_before, 1)
    idle_delta = idle_after - idle_before
    overall = 100.0 * (1.0 - idle_delta / total_delta)

    usage = []
    for pid, process in after.items():
        previous = before.get(pid)
        # A pid that appeared mid-interval has no baseline; charge it from 0 so
        # short-lived spikes still show up rather than disappearing.
        base = previous.cpu_ticks if previous else 0
        ticks = process.cpu_ticks - base
        if ticks < 0:
            continue
        percent = 100.0 * ticks / CLOCK_TICKS / interval
        usage.append((percent, process))
    usage.sort(key=lambda item: item[0], reverse=True)
    return overall, breakdown, usage


def format_bytes(value):
    return f"{value / GIB:.1f}G"


def cpu_budget():
    """(usable cores, physical cores) -- these differ under cgroup/taskset."""

    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 1
    return usable, os.cpu_count() or usable


def build_report(interval, only_user, top, everything=False):
    """Gather everything one report needs, without printing it."""

    overall, breakdown, usage = measure(interval)
    usable, physical = cpu_budget()
    memory = read_meminfo()
    with open("/proc/loadavg") as handle:
        load1, load5, load15 = (float(value) for value in handle.read().split()[:3])

    if not everything:
        # Kernel threads, htop, sshd and friends drown out the thing being
        # diagnosed. The machine-wide CPU/MEM/SWAP lines above still account
        # for them, so nothing is hidden -- only the listings are narrowed.
        me = os.getpid()
        usage = [
            item for item in usage if is_python(item[1]) and item[1].pid != me
        ]

    per_user_cpu = defaultdict(float)
    per_user_rss = defaultdict(int)
    per_user_threads = defaultdict(int)
    for percent, process in usage:
        per_user_cpu[process.user] += percent
        per_user_rss[process.user] += process.rss
        per_user_threads[process.user] += process.threads

    totals = {
        "processes": len(usage),
        "cpu": sum(percent for percent, _ in usage),
        "rss": sum(process.rss for _, process in usage),
        "threads": sum(process.threads for _, process in usage),
    }

    if only_user:
        usage = [item for item in usage if item[1].user == only_user]

    return {
        "overall": overall,
        "breakdown": breakdown,
        "usage": usage,
        "totals": totals,
        "top": top,
        "usable": usable,
        "physical": physical,
        "memory": memory,
        "cgroup_limit": read_cgroup_memory_limit(),
        "load": (load1, load5, load15),
        "per_user_cpu": per_user_cpu,
        "per_user_rss": per_user_rss,
        "per_user_threads": per_user_threads,
        "only_user": only_user,
        "everything": everything,
    }


def print_report(report):
    usable = report["usable"]
    physical = report["physical"]
    load1, load5, load15 = report["load"]
    memory = report["memory"]
    breakdown = report["breakdown"]

    total = memory.get("MemTotal", 0)
    available = memory.get("MemAvailable", 0)
    used = total - available
    swap_total = memory.get("SwapTotal", 0)
    swap_used = swap_total - memory.get("SwapFree", 0)

    cores = f"{usable} cores"
    if usable != physical:
        cores += f" usable of {physical} on the box"

    print("=" * 78)
    print(f"CPU   {report['overall']:5.1f}% busy across {cores}")
    print(
        f"      load {load1:.2f} / {load5:.2f} / {load15:.2f}"
        f"   =  {load1 / usable:.2f} / {load5 / usable:.2f} / {load15 / usable:.2f} per core"
        "   (1m/5m/15m)"
    )

    steal = breakdown.get("steal", 0)
    iowait = breakdown.get("iowait", 0)
    cumulative = max(sum(breakdown.values()), 1)
    print(
        f"      cumulative since boot: iowait {100.0 * iowait / cumulative:.1f}%,"
        f" steal {100.0 * steal / cumulative:.1f}%"
    )

    percent_used = 100.0 * used / total if total else 0.0
    print(
        f"MEM   {format_bytes(used)} used of {format_bytes(total)}"
        f" ({percent_used:.0f}%), {format_bytes(available)} available"
    )
    if swap_total:
        print(
            f"SWAP  {format_bytes(swap_used)} used of {format_bytes(swap_total)}"
            f" ({100.0 * swap_used / swap_total:.0f}%)"
        )
    if report["cgroup_limit"]:
        print(f"LIMIT cgroup memory ceiling {format_bytes(report['cgroup_limit'])}")

    scope = "ALL" if report["everything"] else "PY"
    totals = report["totals"]
    print(
        f"{scope:<6}{totals['processes']} processes,"
        f" {totals['cpu']:.1f}% CPU ({totals['cpu'] / 100.0:.1f} cores),"
        f" {format_bytes(totals['rss'])} RSS,"
        f" {totals['threads']} threads"
    )

    print()
    print(f"{'USER':<14}{'CPU%':>9}{'CORES':>8}{'MEM':>9}{'THREADS':>9}")
    print("-" * 78)
    ranked = sorted(report["per_user_cpu"].items(), key=lambda item: item[1], reverse=True)
    for user, percent in ranked:
        if percent < 1.0 and report["per_user_rss"][user] < GIB:
            continue
        print(
            f"{user:<14}{percent:>8.1f}%{percent / 100.0:>8.1f}"
            f"{format_bytes(report['per_user_rss'][user]):>9}"
            f"{report['per_user_threads'][user]:>9}"
        )
    if not ranked:
        print("(no python processes running)")

    print()
    label = f" for {report['only_user']}" if report["only_user"] else ""
    heading = "PROCESSES" if report["everything"] else "PYTHON PROCESSES"
    print(f"TOP {heading}{label}")
    print(f"{'PID':>8} {'USER':<11}{'CPU%':>9}{'MEM':>8}{'THR':>6}  COMMAND")
    print("-" * 78)
    for percent, process in report["usage"][: report["top"]]:
        command = process.cmdline or f"[{process.comm}]"
        if len(command) > 34:
            command = command[:31] + "..."
        print(
            f"{process.pid:>8} {process.user:<11}{percent:>8.1f}%"
            f"{format_bytes(process.rss):>8}{process.threads:>6}  {command}"
        )

    print()
    for line in verdicts(report):
        print(line)
    print()


def verdicts(report):
    """Plain-language findings. Empty when nothing stands out."""

    usable = report["usable"]
    load1, load5, load15 = report["load"]
    memory = report["memory"]
    lines = []

    total = memory.get("MemTotal", 0)
    available = memory.get("MemAvailable", 0)
    swap_total = memory.get("SwapTotal", 0)
    swap_used = swap_total - memory.get("SwapFree", 0)

    if load15 / usable > 1.5:
        lines.append(
            f"! Load is {load15 / usable:.1f}x the core count over 15 minutes. More work is"
        )
        lines.append(
            "  queued than the machine can run; everyone's jobs get slower even though"
        )
        lines.append("  CPU% may look unremarkable. This is the usual cause of complaints.")
    elif load5 / usable > 1.5 > load15 / usable:
        lines.append(
            f"! Load spiked recently ({load5 / usable:.1f}x cores over 5 minutes) but the"
        )
        lines.append("  15-minute figure is calmer. A snapshot now would have missed it.")

    if total and available / total < 0.10:
        lines.append(
            f"! Only {format_bytes(available)} of {format_bytes(total)} memory is available"
            f" ({100.0 * available / total:.0f}%)."
        )
    if swap_total and swap_used / swap_total > 0.25:
        lines.append(
            f"! Swap is {100.0 * swap_used / swap_total:.0f}% used"
            f" ({format_bytes(swap_used)}). The machine is paging to disk, which slows"
        )
        lines.append("  every user down far more than high CPU does.")

    me = pwd.getpwuid(os.getuid()).pw_name
    my_threads = report["per_user_threads"].get(me, 0)
    scope = "" if report["everything"] else "python "
    if my_threads > 4 * usable:
        lines.append(
            f"! Your {scope}processes hold {my_threads} threads on {usable} cores"
            f" ({my_threads / usable:.0f}x)."
        )
        lines.append(
            "  PyTorch spawns one compute thread per core per process unless capped."
        )
        lines.append(
            "  Set OMP_NUM_THREADS (and torch.set_num_threads) when running several"
        )
        lines.append("  workers or trials at once.")
    else:
        uncapped = []
        for _, process in report["usage"][:20]:
            if process.user != me or process.threads <= usable:
                continue
            if not thread_limits(process.pid):
                uncapped.append(process.pid)
        if uncapped:
            listed = ", ".join(str(pid) for pid in uncapped[:5])
            lines.append(
                f"* Your pids {listed} run more threads than cores with no"
                " OMP_NUM_THREADS set."
            )

    if not lines:
        if not report["totals"]["processes"]:
            lines.append("No python processes running, and the machine looks healthy.")
            lines.append("Pass --all to include everything else that is running.")
            return lines
        lines.append("Nothing unusual in this sample.")
        lines.append(
            "If the complaint is about another time of day, leave --log running:"
        )
        lines.append(
            "  nohup python3 scripts/server_usage.py --watch --interval 60"
            " --log usage.csv &"
        )
    return lines


CSV_COLUMNS = (
    "timestamp", "cpu_percent", "cores", "load1", "load5", "load15",
    "load15_per_core", "mem_used_gib", "mem_total_gib", "mem_available_gib",
    "swap_used_gib", "py_processes", "py_cpu_percent", "py_mem_gib",
    "py_threads", "my_cpu_percent", "my_mem_gib", "my_threads",
)


def log_row(report, path):
    """Append one sample to a CSV, writing the header on first use."""

    memory = report["memory"]
    me = pwd.getpwuid(os.getuid()).pw_name
    total = memory.get("MemTotal", 0)
    available = memory.get("MemAvailable", 0)
    load1, load5, load15 = report["load"]

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": round(report["overall"], 1),
        "cores": report["usable"],
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load15_per_core": round(load15 / report["usable"], 2),
        "mem_used_gib": round((total - available) / GIB, 2),
        "mem_total_gib": round(total / GIB, 2),
        "mem_available_gib": round(available / GIB, 2),
        "swap_used_gib": round(
            (memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)) / GIB, 2
        ),
        "py_processes": report["totals"]["processes"],
        "py_cpu_percent": round(report["totals"]["cpu"], 1),
        "py_mem_gib": round(report["totals"]["rss"] / GIB, 2),
        "py_threads": report["totals"]["threads"],
        "my_cpu_percent": round(report["per_user_cpu"].get(me, 0.0), 1),
        "my_mem_gib": round(report["per_user_rss"].get(me, 0) / GIB, 2),
        "my_threads": report["per_user_threads"].get(me, 0),
    }
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between the two samples a CPU%% reading needs (default: 2)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep sampling until interrupted instead of reporting once",
    )
    parser.add_argument(
        "--top", type=int, default=12, help="how many processes to list (default: 12)"
    )
    parser.add_argument(
        "--me",
        action="store_true",
        help="list only your own processes (per-user totals still cover everyone)",
    )
    parser.add_argument(
        "--user", help="list only this user's processes"
    )
    parser.add_argument(
        "--all",
        dest="everything",
        action="store_true",
        help="include non-python processes and kernel threads (default: python only)",
    )
    parser.add_argument(
        "--log",
        metavar="CSV",
        help="append each sample to a CSV file, for catching usage you are not watching",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write to --log without printing a report",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not sys.platform.startswith("linux"):
        print("This script reads /proc and only works on Linux.", file=sys.stderr)
        return 2
    if args.quiet and not args.log:
        print("--quiet only makes sense together with --log.", file=sys.stderr)
        return 2

    only_user = args.user
    if args.me:
        only_user = pwd.getpwuid(os.getuid()).pw_name

    try:
        while True:
            report = build_report(
                args.interval, only_user, args.top, everything=args.everything
            )
            if args.log:
                log_row(report, args.log)
            if not args.quiet:
                if args.watch:
                    # Clear and home, so a watched terminal refreshes in place.
                    print("\033[2J\033[H", end="")
                print_report(report)
            if not args.watch:
                return 0
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
