import os
import sys
import threading
import time

import psutil
import pytest

from marble.launcher import LocalLauncher, Slot, default_slots
from marble.spec import Resources

PY = sys.executable


def test_successful_job_and_env(launcher, tmp_path):
    log = tmp_path / "log.txt"
    r = launcher.run(f"{PY} -c \"import os; print(os.environ['OMP_NUM_THREADS'], repr(os.environ['CUDA_VISIBLE_DEVICES']))\"",
                     cwd=tmp_path, resources=Resources(cpu=1), timeout=30, log_path=log)
    assert r.ok and r.returncode == 0
    assert log.read_text().strip() == "2 ''"   # CPU slot hides GPUs, threads = slot cpus
    assert r.duration_s < 5


def test_nonzero_exit(launcher, tmp_path):
    r = launcher.run("exit 3", cwd=tmp_path, resources=Resources(), timeout=30, log_path=tmp_path / "l")
    assert r.returncode == 3 and not r.ok


def test_timeout_kills_whole_tree(launcher, tmp_path):
    # Parent spawns a grandchild that ignores SIGTERM; both must be gone after the timeout.
    pidfile = tmp_path / "child.pid"
    script = (f"import os, signal, subprocess, sys, time\n"
              f"p = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)'])\n"
              f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
              f"time.sleep(600)\n")
    (tmp_path / "s.py").write_text(script)
    t = time.monotonic()
    r = launcher.run(f"{PY} s.py", cwd=tmp_path, resources=Resources(), timeout=1.0, log_path=tmp_path / "l")
    assert r.timed_out and not r.ok
    assert time.monotonic() - t < 20
    child = int(pidfile.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_cancel_and_probe(launcher, tmp_path):
    slot = launcher.acquire(Resources())
    try:
        h = launcher.start(f"{PY} -c \"import time\nwhile True: pass\"", cwd=tmp_path, slot=slot,
                           timeout=60, log_path=tmp_path / "l")
        time.sleep(1.0)
        s = h.probe()
        assert s.alive and s.cpu_percent > 30          # busy loop is visibly using CPU
        h.cancel("test")
        r = h.wait(30)
        assert r.cancelled and r.cancel_reason == "test"
    finally:
        launcher.release(slot)


def test_idle_job_shows_no_output_and_no_cpu(launcher, tmp_path):
    seen = []
    r = launcher.run(f"{PY} -c \"print('start', flush=True); import time; time.sleep(1.5)\"", cwd=tmp_path,
                     resources=Resources(), timeout=30, log_path=tmp_path / "l",
                     on_sample=lambda h, s: seen.append(s))
    assert r.ok
    late = [s for s in seen if s.t > 1.0 and s.alive]
    assert late and all(s.cpu_percent < 20 for s in late)
    assert max(s.seconds_since_output for s in late) > 0.5


def test_slot_exclusivity(tmp_path):
    lch = LocalLauncher([Slot("only", cpus=1)], sample_interval=0.1)
    order = []

    def job(i):
        lch.run(f"{PY} -c \"import time; time.sleep(0.5)\"", cwd=tmp_path, resources=Resources(),
                timeout=30, log_path=tmp_path / f"l{i}",
                on_sample=lambda h, s, i=i: order.append(i) if not order or order[-1] != i else None)

    ts = [threading.Thread(target=job, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(order) == [0, 1]   # one job at a time: samples are not interleaved


def test_no_fitting_slot(launcher):
    with pytest.raises(RuntimeError, match="no slot"):
        launcher.acquire(Resources(gpu=1))


def test_default_slots_pinning():
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 4:
        pytest.skip("needs >= 4 CPUs")
    slots = default_slots(gpus=[], cpu_slots=2, cpus_per_slot=2, pin=True)
    assert len(slots) == 2 and set(slots[0].cpu_ids).isdisjoint(slots[1].cpu_ids)
