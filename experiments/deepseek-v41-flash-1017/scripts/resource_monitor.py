#!/usr/bin/env python3
"""Sample GPU, host and Docker cgroup-v2 memory; no privileged writes."""
import argparse
import csv
import json
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    return subprocess.check_output(args, text=True, timeout=10, stderr=subprocess.DEVNULL)


def scalar(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pairs(path):
    try:
        return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def container_sample(name):
    try:
        state = json.loads(command(["docker", "inspect", "--format", "{{json .State}}", name]))
        result = {"container_running": state["Running"], "container_oom_killed": state["OOMKilled"]}
        pid = state["Pid"]
        if not pid:
            return result
        rel = next(line.split("::", 1)[1] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
        group = Path("/sys/fs/cgroup") / rel.lstrip("/")
        result["cgroup_path"] = str(group)
        for key, file in (("container_memory_current_bytes", "memory.current"),
                          ("container_memory_peak_bytes", "memory.peak"),
                          ("container_swap_bytes", "memory.swap.current")):
            result[key] = scalar(group / file)
        stats = pairs(group / "memory.stat")
        for key in ("anon", "file", "shmem", "kernel"):
            result[f"container_{key}_bytes"] = stats.get(key)
        result["cgroup_memory_events"] = pairs(group / "memory.events")
        rss = []
        for item in (group / "cgroup.procs").read_text().split():
            try:
                for line in Path(f"/proc/{int(item)}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss.append(int(line.split()[1]) * 1024)
            except OSError:
                pass
        result["process_rss_sum_bytes_shared_pages_counted_multiple_times"] = sum(rss)
        result["process_rss_count"] = len(rss)
        return result
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        return {"container_sampling_error": type(error).__name__}


def sample(args):
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "unix_s": time.time()}
    try:
        row["phase"] = json.loads(args.state.read_text())["stage"]
    except (OSError, KeyError, ValueError):
        row["phase"] = "unknown"
    mem = {k: int(v.split()[0]) * 1024 for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    for key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "AnonPages", "Shmem", "Slab", "SwapTotal", "SwapFree"):
        row[f"host_{key}_bytes"] = mem.get(key)
    fields = ("memory.used", "memory.total", "utilization.gpu", "power.draw", "temperature.gpu")
    values = next(csv.reader(command(["nvidia-smi", "--id=0", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]).splitlines()))
    for key, value in zip(("gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_utilization_pct", "gpu_power_w", "gpu_temperature_c"), values):
        try:
            row[key] = float(value.strip())
        except ValueError:
            row[key] = None
    row.update(container_sample(args.container))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=1)
    ap.add_argument("--max-seconds", type=int, default=14400)
    args = ap.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    deadline = time.monotonic() + args.max_seconds
    with args.output.open("w", buffering=1) as output:
        while not stop.is_set() and time.monotonic() < deadline:
            start = time.monotonic()
            try:
                row = sample(args)
            except Exception as error:
                row = {"time_utc": datetime.now(timezone.utc).isoformat(), "sampling_error": repr(error)}
            output.write(json.dumps(row) + "\n")
            stop.wait(max(0, args.interval - (time.monotonic() - start)))
    if not stop.is_set():
        raise SystemExit("Resource monitor reached its time limit")


if __name__ == "__main__":
    main()
