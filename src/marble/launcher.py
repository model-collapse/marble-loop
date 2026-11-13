"""Local launcher: resource slots, process-group jobs, hard timeouts, live probes.

Every job runs in its own session/process group so that a timeout or cancel kills the whole tree.
A monitor thread samples the tree (CPU, RSS, log growth, GPU utilisation) so the Inspector can tell
"slow but working" from "hung" (see DESIGN.md §6).
"""

from __future__ import annotations

import collections
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import psutil

from .spec import Resources

KILL_GRACE_S = 10.0


# --------------------------------------------------------------------------- slots


@dataclass(frozen=True)
class Slot:
    name: str
    gpus: tuple[int, ...] = ()
    cpus: int = 1
    cpu_ids: tuple[int, ...] = ()  # optional CPU pinning (reduces timing noise)

    def fits(self, r: Resources) -> bool:
        return len(self.gpus) >= r.gpu and self.cpus >= r.cpu

    def env(self) -> dict[str, str]:
        threads = str(self.cpus)
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in self.gpus),
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "MARBLE_SLOT": self.name,
        }


def detect_gpus() -> list[int]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        return [int(x) for x in out.split()]
    except (subprocess.SubprocessError, ValueError):
        return []


def default_slots(gpus: list[int] | None = None, cpu_slots: int = 1, cpus_per_slot: int | None = None,
                  pin: bool = False) -> list[Slot]:
    """One slot per GPU (each with a share of the CPUs) plus `cpu_slots` CPU-only slots.

    With pin=True each slot gets a disjoint contiguous range of CPU ids (less timing noise).
    """
    gpus = detect_gpus() if gpus is None else gpus
    avail = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    n = len(gpus) + cpu_slots
    per = cpus_per_slot or max(1, len(avail) // max(1, n))
    if pin and per * n > len(avail):
        raise ValueError(f"cannot pin {n} slots x {per} CPUs on {len(avail)} available CPUs")
    ranges = [tuple(avail[i * per:(i + 1) * per]) if pin else () for i in range(n)]
    slots = [Slot(f"gpu{g}", gpus=(g,), cpus=per, cpu_ids=ranges[i]) for i, g in enumerate(gpus)]
    slots += [Slot(f"cpu{i}", cpus=per, cpu_ids=ranges[len(gpus) + i]) for i in range(cpu_slots)]
    return slots


# --------------------------------------------------------------------------- probes


@dataclass
class Sample:
    t: float  # seconds since job start
    alive: bool
    num_procs: int
    cpu_percent: float  # summed over the process tree, 100 = one full core
    rss_bytes: int
    log_bytes: int
    seconds_since_output: float
    gpu_util: list[int] | None = None  # utilisation (%) of the slot's GPUs


def _gpu_util(gpus: tuple[int, ...]) -> list[int] | None:
    if not gpus:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return [int(x) for x in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


# --------------------------------------------------------------------------- jobs


@dataclass
class JobResult:
    returncode: int | None
    duration_s: float
    timed_out: bool
    cancelled: bool
    cancel_reason: str | None
    cpu_seconds: float
    peak_rss_bytes: int
    last_output_age_s: float
    log_path: str
    slot: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled

    def to_dict(self) -> dict:
        return asdict(self)


class JobHandle:
    def __init__(self, proc: subprocess.Popen, log_path: Path, slot: Slot, timeout: float,
                 sample_interval: float, on_sample: Callable[["JobHandle", Sample], None] | None):
        self.proc = proc
        self.log_path = log_path
        self.slot = slot
        self.timeout = timeout
        self.start = time.monotonic()
        self.samples: collections.deque[Sample] = collections.deque(maxlen=600)
        self._on_sample = on_sample
        self._interval = sample_interval
        self._cancel_reason: str | None = None
        self._timed_out = False
        self._done = threading.Event()
        self._cpu_by_pid: dict[int, float] = {}
        self._cpu_total = 0.0
        self._peak_rss = 0
        self._last_log_size = 0
        self._last_output_t = self.start
        self._result: JobResult | None = None
        self._monitor = threading.Thread(target=self._run_monitor, name=f"marble-mon-{proc.pid}", daemon=True)
        self._monitor.start()

    # -- public API
    @property
    def pid(self) -> int:
        return self.proc.pid

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        if self._cancel_reason is None and not self.done():
            self._cancel_reason = reason
            self._kill_tree()

    def probe(self) -> Sample:
        """Take a fresh sample now (also used by the monitor thread)."""
        return self._sample(gpu=True)

    def wait(self, timeout: float | None = None) -> JobResult:
        if not self._done.wait(timeout):
            raise TimeoutError("job still running")
        assert self._result is not None
        return self._result

    # -- internals
    def _tree(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.proc.pid)
            return [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def _sample(self, gpu: bool = False) -> Sample:
        now = time.monotonic()
        procs = self._tree()
        rss, cpu_now = 0, {}
        for p in procs:
            try:
                with p.oneshot():
                    ct = p.cpu_times()
                    cpu_now[p.pid] = ct.user + ct.system
                    rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        delta = sum(max(0.0, v - self._cpu_by_pid.get(pid, 0.0)) for pid, v in cpu_now.items())
        prev_t = getattr(self, "_last_sample_t", self.start)
        self._last_sample_t = now
        self._cpu_by_pid.update(cpu_now)
        self._cpu_total += delta
        self._peak_rss = max(self._peak_rss, rss)
        try:
            size = self.log_path.stat().st_size
        except OSError:
            size = self._last_log_size
        if size != self._last_log_size:
            self._last_log_size = size
            self._last_output_t = now
        dt = max(1e-6, now - prev_t)
        return Sample(
            t=now - self.start, alive=self.proc.poll() is None, num_procs=len(procs),
            cpu_percent=100.0 * delta / dt, rss_bytes=rss, log_bytes=size,
            seconds_since_output=now - self._last_output_t,
            gpu_util=_gpu_util(self.slot.gpus) if gpu else None,
        )

    def _run_monitor(self) -> None:
        n = 0
        while self.proc.poll() is None:
            if self.elapsed() > self.timeout and not self._timed_out and self._cancel_reason is None:
                self._timed_out = True
                self._kill_tree()
            s = self._sample(gpu=(n % 5 == 0))
            self.samples.append(s)
            if self._on_sample is not None:
                try:
                    self._on_sample(self, s)
                except Exception:  # an observer must never kill the job
                    pass
            n += 1
            try:  # returns as soon as the process exits, so durations are not quantised
                self.proc.wait(timeout=self._interval)
            except subprocess.TimeoutExpired:
                pass
        end = time.monotonic()
        final = self._sample()
        self.samples.append(final)
        self._result = JobResult(
            returncode=self.proc.returncode, duration_s=end - self.start, timed_out=self._timed_out,
            cancelled=self._cancel_reason is not None, cancel_reason=self._cancel_reason,
            cpu_seconds=self._cpu_total, peak_rss_bytes=self._peak_rss,
            last_output_age_s=final.seconds_since_output, log_path=str(self.log_path), slot=self.slot.name,
        )
        self._done.set()

    def _kill_tree(self) -> None:
        procs = self._tree()
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + KILL_GRACE_S
        while time.monotonic() < deadline and any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in procs if _alive(p)):
            time.sleep(0.2)
        for p in procs:  # also catches children that left the process group
            if _alive(p):
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _alive(p: psutil.Process) -> bool:
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class LocalLauncher:
    """Runs shell commands on local slots; one job per slot at a time."""

    def __init__(self, slots: list[Slot], sample_interval: float = 1.0):
        if not slots:
            raise ValueError("launcher needs at least one slot")
        self.slots = list(slots)
        self.sample_interval = sample_interval
        self._busy: set[str] = set()
        self._cv = threading.Condition()

    def acquire(self, resources: Resources, timeout: float | None = None) -> Slot:
        candidates = [s for s in self.slots if s.fits(resources)]
        if not candidates:
            raise RuntimeError(f"no slot can satisfy {resources} (slots: {[s.name for s in self.slots]})")
        # Prefer the smallest fitting slot so CPU jobs don't occupy GPU slots.
        candidates.sort(key=lambda s: (len(s.gpus), s.cpus))
        with self._cv:
            ok = self._cv.wait_for(lambda: any(s.name not in self._busy for s in candidates), timeout)
            if not ok:
                raise TimeoutError("no free slot")
            slot = next(s for s in candidates if s.name not in self._busy)
            self._busy.add(slot.name)
            return slot

    def release(self, slot: Slot) -> None:
        with self._cv:
            self._busy.discard(slot.name)
            self._cv.notify_all()

    def start(self, cmd: str, *, cwd: Path, slot: Slot, timeout: float, log_path: Path,
              env: dict[str, str] | None = None,
              on_sample: Callable[[JobHandle, Sample], None] | None = None) -> JobHandle:
        full_env = dict(os.environ)
        full_env.update(env or {})
        full_env.update(slot.env())
        if slot.cpu_ids and shutil.which("taskset"):
            cmd = f"taskset -c {','.join(map(str, slot.cpu_ids))} bash -c {shlex.quote(cmd)}"
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                ["bash", "-c", cmd], cwd=str(cwd), env=full_env, stdin=subprocess.DEVNULL,
                stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
            )
        finally:
            log_f.close()
        return JobHandle(proc, log_path, slot, timeout, self.sample_interval, on_sample)

    def run(self, cmd: str, *, cwd: Path, resources: Resources, timeout: float, log_path: Path,
            env: dict[str, str] | None = None,
            on_sample: Callable[[JobHandle, Sample], None] | None = None) -> JobResult:
        """Acquire a slot, run to completion (or timeout), release."""
        slot = self.acquire(resources)
        try:
            h = self.start(cmd, cwd=cwd, slot=slot, timeout=timeout, log_path=log_path, env=env, on_sample=on_sample)
            return h.wait()
        finally:
            self.release(slot)
