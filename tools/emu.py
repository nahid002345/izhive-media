#!/usr/bin/env python3
"""Claude-side client for the on-demand Prism emulator.

  python3 emu.py start [--max 30] [--idle 12]     -> boots a session, prints its id
  python3 emu.py run SID actions.json|'<json>' [--out DIR]
  python3 emu.py stop SID
  python3 emu.py status SID

Sessions run on GitHub Actions (repo nahid002345/izhive-media):
control branch  prism-emulator          (workflow, tools, control/start.json)
bridge branch   prism-emulator-bridge   (sessions/<sid>/commands, results, status)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_URL = "https://github.com/nahid002345/izhive-media"
CONTROL = "prism-emulator"
BRIDGE = "prism-emulator-bridge"
HOME = Path("/home/claude/.prism-emu")
CTRL_DIR, BR_DIR = HOME / "control", HOME / "bridge"


def git(args, cwd, check=True, timeout=300):
    r = subprocess.run(f"git {args}", shell=True, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"git {args} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def ensure_clone(path, branch):
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git clone -q --depth 1 --branch {branch} {REPO_URL} {path}", shell=True, check=True,
                       timeout=600)
        git('config user.name "Claude"', path)
        git('config user.email "noreply@anthropic.com"', path)


def sync(path, branch):
    git(f"fetch -q --depth 1 origin {branch}", path)
    git(f"reset -q --hard origin/{branch}", path)


def push(path, branch, msg):
    git("add -A", path)
    git(f'commit -q -m "{msg}"', path, check=False)
    for _ in range(5):
        r = subprocess.run(f"git push -q origin HEAD:{branch}", shell=True, cwd=path, text=True, capture_output=True)
        if r.returncode == 0:
            return
        git(f"fetch -q --depth 1 origin {branch}", path)
        git(f"rebase -q origin/{branch}", path, check=False)
        time.sleep(2)
    raise RuntimeError("push failed: " + r.stderr)


def status_of(sid):
    sync(BR_DIR, BRIDGE)
    p = BR_DIR / "sessions" / sid / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def wait_status(sid, states, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = status_of(sid)
        if st and st["state"] in states:
            return st
        time.sleep(10)
    return None


def copy_results(sid, cid, out):
    src = BR_DIR / "sessions" / sid / "results" / cid
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, out / f.name)
    return out


def cmd_start(a):
    ensure_clone(CTRL_DIR, CONTROL)
    ensure_clone(BR_DIR, BRIDGE)
    sync(CTRL_DIR, CONTROL)
    sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    (CTRL_DIR / "control").mkdir(exist_ok=True)
    (CTRL_DIR / "control/start.json").write_text(json.dumps(
        {"session": sid, "max_minutes": a.max, "idle_minutes": a.idle,
         "requested": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    push(CTRL_DIR, CONTROL, f"start emulator session {sid}")
    print(f"session {sid} requested; waiting for boot (usually 4-8 min)...", flush=True)
    st = wait_status(sid, {"ready", "stopped"}, a.timeout)
    print(json.dumps(st, indent=2) if st else "timeout waiting for session")
    if st and st["state"] == "ready":
        out = copy_results(sid, "0000", Path(a.out) / sid / "0000")
        print("launch files:", out)
    print("SID=" + sid)


def next_id(sid):
    cdir = BR_DIR / "sessions" / sid / "commands"
    nums = [int(p.stem) for p in cdir.glob("*.json")] if cdir.exists() else []
    return f"{(max(nums) + 1) if nums else 1:04d}"


def cmd_run(a):
    ensure_clone(BR_DIR, BRIDGE)
    sync(BR_DIR, BRIDGE)
    payload = a.actions
    actions = json.loads(Path(payload).read_text() if Path(payload).exists() else payload)
    cid = next_id(a.sid)
    cdir = BR_DIR / "sessions" / a.sid / "commands"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{cid}.json").write_text(json.dumps(actions, indent=2))
    push(BR_DIR, BRIDGE, f"session {a.sid}: command {cid}")
    t0 = time.time()
    done = BR_DIR / "sessions" / a.sid / "results" / cid / "done.json"
    while time.time() - t0 < a.timeout:
        time.sleep(4)
        sync(BR_DIR, BRIDGE)
        if done.exists():
            out = copy_results(a.sid, cid, Path(a.out) / a.sid / cid)
            print(done.read_text())
            print("files in", out)
            return
        st = status_of(a.sid)
        if st and st["state"] == "stopped":
            print("session stopped:", st)
            return
    print("timeout waiting for result", cid)


def cmd_stop(a):
    a.actions = json.dumps([{"do": "stop_session"}])
    a.timeout = 120
    cmd_run(a)


def cmd_status(a):
    ensure_clone(BR_DIR, BRIDGE)
    print(json.dumps(status_of(a.sid), indent=2))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("start"); s.add_argument("--max", type=float, default=30); s.add_argument("--idle", type=float, default=12)
s.add_argument("--timeout", type=int, default=900); s.add_argument("--out", default="/home/claude/emu-out"); s.set_defaults(f=cmd_start)
r = sub.add_parser("run"); r.add_argument("sid"); r.add_argument("actions"); r.add_argument("--out", default="/home/claude/emu-out")
r.add_argument("--timeout", type=int, default=300); r.set_defaults(f=cmd_run)
t = sub.add_parser("stop"); t.add_argument("sid"); t.add_argument("--out", default="/home/claude/emu-out"); t.set_defaults(f=cmd_stop)
u = sub.add_parser("status"); u.add_argument("sid"); u.set_defaults(f=cmd_status)
args = ap.parse_args()
args.f(args)
