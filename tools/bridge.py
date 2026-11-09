#!/usr/bin/env python3
"""Command bridge between Claude and an Android emulator on a GitHub runner.

Claude writes sessions/<sid>/commands/<id>.json on the prism-emulator-bridge
branch; this loop runs the actions with adb and pushes
sessions/<sid>/results/<id>/ (screenshots, UI dumps, done.json) back.
"""
import json
import os
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PKG = "com.nahid002345.prismunitconverter"
BRIDGE = "prism-emulator-bridge"
ROOT = Path.cwd()
START = json.loads((ROOT / "control/start.json").read_text())
SID = START["session"]
MAX_MIN = float(START.get("max_minutes", 30))
IDLE_MIN = float(START.get("idle_minutes", 12))
WT = Path("/tmp/bridge")
T0 = time.time()


def run(cmd, cwd=None, timeout=180):
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def adb(args, timeout=60):
    return run(f"adb {args}", timeout=timeout)


# ---------------------------------------------------------------- git bridge
def git_setup():
    token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    code, out = run(f"git clone -q --depth 1 --branch {BRIDGE} {url} {WT}")
    if code:
        WT.mkdir(parents=True, exist_ok=True)
        run("git init -q && git checkout -q --orphan " + BRIDGE, cwd=WT)
        run(f"git remote add origin {url}", cwd=WT)
    run('git config user.name "prism-emulator" && git config user.email "actions@users.noreply.github.com"', cwd=WT)


def sdir():
    d = WT / "sessions" / SID
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d


def push(msg):
    run(f"git add -A && git commit -q -m {shlex.quote(msg)}", cwd=WT)
    for _ in range(6):
        code, out = run(f"git push -q origin HEAD:{BRIDGE}", cwd=WT)
        if code == 0:
            return True
        run(f"git pull -q --rebase -X theirs origin {BRIDGE}", cwd=WT)
        time.sleep(1)
    print("push failed", out, flush=True)
    return False


def pull():
    run(f"git pull -q --rebase -X theirs origin {BRIDGE}", cwd=WT, timeout=60)


def status(state, **extra):
    data = {"session": SID, "state": state, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run": f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"}
    data.update(extra)
    (sdir() / "status.json").write_text(json.dumps(data, indent=2))
    push(f"session {SID}: {state}")


# ------------------------------------------------------------------- device
def dump_ui():
    adb("shell uiautomator dump /sdcard/ui.xml", timeout=40)
    code, xml = adb("exec-out cat /sdcard/ui.xml")
    return xml if xml.strip().startswith("<?xml") else ""


def nodes(xml):
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for n in root.iter("node"):
        b = re.findall(r"\d+", n.get("bounds", ""))
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = map(int, b)
        out.append({"text": n.get("text", ""), "desc": n.get("content-desc", ""), "id": n.get("resource-id", ""),
                    "cls": n.get("class", "").split(".")[-1], "click": n.get("clickable") == "true",
                    "edit": "EditText" in n.get("class", ""), "focused": n.get("focused") == "true",
                    "bounds": [x1, y1, x2, y2], "center": [(x1 + x2) // 2, (y1 + y2) // 2]})
    return out


def summary(ns):
    lines = []
    for n in ns:
        label = n["text"] or n["desc"] or n["id"]
        if label or n["click"] or n["edit"]:
            flags = ("C" if n["click"] else "-") + ("E" if n["edit"] else "-")
            lines.append(f'{flags} {n["cls"]:<14} {n["center"]} {n["bounds"]} text={n["text"]!r} desc={n["desc"]!r} id={n["id"]!r}')
    return "\n".join(lines)


def find(ns, a):
    want = a.get("text") or a.get("desc")
    rid = a.get("id")
    contains = a.get("contains", False)
    nth = a.get("nth", 0)
    hits = []
    for n in ns:
        if rid and n["id"].endswith(rid):
            hits.append(n)
            continue
        if want:
            for field in (n["text"], n["desc"]):
                if (contains and want.lower() in field.lower()) or field == want:
                    hits.append(n)
                    break
    if a.get("edit_only"):
        hits = [h for h in hits if h["edit"]]
    return hits[nth] if len(hits) > nth else None


def type_text(text):
    # adb input text: spaces as %s, keep it ASCII; quote for the device shell
    safe = text.replace(" ", "%s")
    return adb(f"shell input text {shlex.quote(shlex.quote(safe))}")


def install():
    info_path = ROOT / "apk/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    split_dir = ROOT / "apk/splits"
    apks = sorted(split_dir.rglob("*.apk")) if split_dir.exists() else []
    if not apks:
        return {"ok": False, "error": "no apk", "info": info}
    _, abilist = adb("shell getprop ro.product.cpu.abilist")
    abis = [a.strip().replace("-", "_") for a in abilist.strip().split(",") if a.strip()]
    abi_splits = [p for p in apks if re.search(r"config\.(arm64_v8a|armeabi_v7a|x86_64|x86)\.apk$", p.name)]
    chosen_abi = None
    for abi in abis:
        for p in abi_splits:
            if p.name == f"config.{abi}.apk":
                chosen_abi = p
                break
        if chosen_abi:
            break
    keep = [p for p in apks if p not in abi_splits] + ([chosen_abi] if chosen_abi else [])
    files = " ".join(shlex.quote(str(p)) for p in keep)
    cmd = "install -r -g " + files if len(keep) == 1 else "install-multiple -r -g " + files
    code, out = adb(cmd, timeout=300)
    return {"ok": code == 0 and "Success" in out, "output": out[-2000:], "device_abis": abis,
            "installed": [p.name for p in keep], "info": {k: info.get(k) for k in ("source", "file", "size_mb", "manifest")}}


def demo_mode(on=True):
    if not on:
        adb("shell am broadcast -a com.android.systemui.demo -e command exit")
        return
    adb("shell settings put global sysui_demo_allowed 1")
    for c in ["-e command enter", "-e command clock -e hhmm 0941",
              "-e command battery -e level 100 -e plugged false",
              "-e command network -e wifi show -e level 4 -e mobile show -e datatype none -e level 4",
              "-e command notifications -e visible false"]:
        adb(f"shell am broadcast -a com.android.systemui.demo {c}")


# ------------------------------------------------------------------ actions
def execute(cmd_id, actions):
    rdir = sdir() / "results" / cmd_id
    rdir.mkdir(parents=True, exist_ok=True)
    log, files = [], []

    def snap(name):
        p = rdir / f"{name}.png"
        run(f"adb exec-out screencap -p > {shlex.quote(str(p))}")
        files.append(p.name)

    def dump(name):
        xml = dump_ui()
        (rdir / f"{name}.xml").write_text(xml)
        (rdir / f"{name}.txt").write_text(summary(nodes(xml)))
        files.extend([f"{name}.xml", f"{name}.txt"])

    for i, a in enumerate(actions):
        do = a.get("do")
        entry = {"i": i, "do": do}
        try:
            if do == "launch":
                adb(f"shell monkey -p {PKG} -c android.intent.category.LAUNCHER 1")
            elif do == "force_stop":
                adb(f"shell am force-stop {PKG}")
            elif do == "clear_data":
                adb(f"shell pm clear {PKG}")
            elif do == "wait":
                time.sleep(a.get("ms", 1000) / 1000)
            elif do == "tap":
                if "x" in a:
                    x, y = a["x"], a["y"]
                else:
                    n = find(nodes(dump_ui()), a)
                    if not n:
                        entry["error"] = "target not found"
                        log.append(entry)
                        if a.get("required", True):
                            break
                        continue
                    x, y = n["center"]
                adb(f"shell input tap {x} {y}")
                entry["at"] = [x, y]
            elif do == "long_press":
                n = find(nodes(dump_ui()), a) if "x" not in a else {"center": [a["x"], a["y"]]}
                x, y = n["center"]
                adb(f"shell input swipe {x} {y} {x} {y} 800")
            elif do == "type":
                if a.get("clear"):
                    adb("shell input keyevent KEYCODE_MOVE_END")
                    adb("shell input keyevent --longpress " + " ".join(["KEYCODE_DEL"] * 60))
                code, out = type_text(a["text"])
                entry["out"] = out.strip()[:200]
            elif do == "key":
                adb(f"shell input keyevent {a['key']}")
            elif do == "swipe":
                adb(f"shell input swipe {a['x1']} {a['y1']} {a['x2']} {a['y2']} {a.get('ms', 300)}")
            elif do == "wait_for":
                deadline = time.time() + a.get("timeout_ms", 15000) / 1000
                found = None
                while time.time() < deadline and not found:
                    found = find(nodes(dump_ui()), a)
                    if not found:
                        time.sleep(1)
                entry["found"] = bool(found)
            elif do == "screenshot":
                snap(a.get("name", f"shot{i}"))
            elif do == "dump":
                dump(a.get("name", f"ui{i}"))
            elif do == "demo_mode":
                demo_mode(a.get("on", True))
            elif do == "shell":
                code, out = adb("shell " + a["cmd"])
                entry["out"] = out[-1500:]
            elif do == "screenrecord":
                secs = int(a.get("seconds", 10))
                name = a.get("name", f"rec{i}")
                adb(f"shell screenrecord --time-limit {secs} /sdcard/{name}.mp4", timeout=secs + 30)
                run(f"adb pull /sdcard/{name}.mp4 {shlex.quote(str(rdir / (name + '.mp4')))}")
                files.append(f"{name}.mp4")
            elif do == "stop_session":
                entry["stop"] = True
            else:
                entry["error"] = f"unknown action {do}"
        except Exception as e:  # keep the session alive
            entry["error"] = repr(e)
        log.append(entry)
        time.sleep(a.get("after_ms", 400) / 1000)

    if actions and actions[-1].get("do") != "screenshot":
        snap("final")
        dump("final")
    _, focus = adb("shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp' | head -2")
    done = {"id": cmd_id, "log": log, "files": files, "focus": focus.strip(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (rdir / "done.json").write_text(json.dumps(done, indent=2))
    push(f"session {SID}: result {cmd_id}")
    return any(e.get("stop") for e in log)


def main():
    git_setup()
    status("booted")
    inst = install()
    demo_mode(True)
    adb(f"shell monkey -p {PKG} -c android.intent.category.LAUNCHER 1")
    time.sleep(8)
    d = sdir() / "results" / "0000"
    d.mkdir(parents=True, exist_ok=True)
    run(f"adb exec-out screencap -p > {d / 'launch.png'}")
    xml = dump_ui()
    (d / "launch.xml").write_text(xml)
    (d / "launch.txt").write_text(summary(nodes(xml)))
    _, model = adb("shell getprop ro.product.model")
    _, sdk = adb("shell getprop ro.build.version.sdk")
    _, ver = adb(f"shell dumpsys package {PKG} | grep -m1 versionName")
    status("ready", install=inst, device=model.strip(), sdk=sdk.strip(), app_version=ver.strip(),
           max_minutes=MAX_MIN, idle_minutes=IDLE_MIN)

    last = time.time()
    while True:
        if time.time() - T0 > MAX_MIN * 60:
            status("stopped", reason="max_minutes reached")
            return
        if time.time() - last > IDLE_MIN * 60:
            status("stopped", reason="idle timeout")
            return
        pull()
        cdir = sdir() / "commands"
        pending = []
        if cdir.exists():
            for f in sorted(cdir.glob("*.json")):
                if not (sdir() / "results" / f.stem / "done.json").exists():
                    pending.append(f)
        for f in pending:
            last = time.time()
            try:
                actions = json.loads(f.read_text())
                actions = actions.get("actions", actions) if isinstance(actions, dict) else actions
            except Exception as e:
                actions = []
                print("bad command", f, e, flush=True)
            if execute(f.stem, actions):
                status("stopped", reason="stop_session command")
                return
        time.sleep(2)


if __name__ == "__main__":
    main()
